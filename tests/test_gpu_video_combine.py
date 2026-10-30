import sys
from types import ModuleType

import pytest
import torch

from nodes.gpu_video_combine import MAIGPUVideoCombine
import nodes.gpu_video_combine as node_module


def test_contract_and_optional_audio():
    node = MAIGPUVideoCombine()
    assert node.CATEGORY == "mAI / IO" and node.OUTPUT_NODE
    assert node.RETURN_TYPES == ("VHS_FILENAMES","STRING","INT","FLOAT")
    assert node.RETURN_NAMES == ("filenames","file_path","frame_count","duration")
    inputs = node.INPUT_TYPES()
    assert inputs["optional"]["audio"] == ("AUDIO",)
    assert inputs["optional"]["h264_quality_preset"][0] == ["custom","low","medium","high"]
    assert inputs["optional"]["h264_quality_preset"][1]["default"] == "custom"
    # Append the new widget after existing ones so positional workflow values stay valid.
    widgets = [name for group in ("required","optional") for name,spec in inputs[group].items()
               if isinstance(spec[0],list) or spec[0] in ("FLOAT","INT","STRING","BOOLEAN")]
    assert widgets == ["frame_rate","filename_prefix","codec","quality","preset","pingpong",
                       "loop_count","trim_to_audio","save_output","save_metadata","gpu_device",
                       "chunk_size","h264_quality_preset"]
    assert inputs["required"]["codec"][1]["default"] == "h264_nvenc"
    assert inputs["required"]["codec"][0] == ["h264_nvenc","hevc_nvenc","av1_nvenc","libx264","libx265","libsvtav1"]
    assert inputs["hidden"] == {"prompt":"PROMPT","extra_pnginfo":"EXTRA_PNGINFO"}


@pytest.mark.parametrize("save_output", [False,True])
@pytest.mark.parametrize("save_metadata", [False,True])
@pytest.mark.parametrize("codec", ["h264_nvenc","libx264"])
def test_save_paths_metadata_preview_and_registration_friendly_output(tmp_path,monkeypatch,save_output,save_metadata,codec):
    folders = ModuleType("folder_paths")
    folders.get_output_directory = lambda:str(tmp_path/"output")
    folders.get_temp_directory = lambda:str(tmp_path/"temp")
    folders.get_save_image_path = lambda prefix,base,w,h:(str(tmp_path/"output" if save_output else tmp_path/"temp"),"mAI",1,"video",prefix)
    monkeypatch.setitem(sys.modules,"folder_paths",folders)
    calls = []
    def encode(frames,path,options,audio,metadata,**kwargs):
        calls.append((path,options,audio,metadata))
        return 121,121/24
    monkeypatch.setattr(node_module,"encode_video",encode)
    workflow = {"nodes":[]}
    frames = torch.zeros(121,4,8,3)
    result = MAIGPUVideoCombine().run(frames,codec=codec,save_output=save_output,save_metadata=save_metadata,
                                     prompt={"test":{}},extra_pnginfo={"workflow":workflow})
    filenames,path,count,duration = result["result"]
    assert filenames == (save_output,[path]) and (count,duration) == (121,121/24)
    assert calls[0][2] is None
    assert calls[0][1].codec == codec
    assert calls[0][1].quality == 23 and calls[0][1].preset == "p4"
    assert calls[0][3] == ({"prompt":{"test":{}},"workflow":workflow} if save_metadata else {})
    preview = result["ui"]["images"][0]
    assert preview["type"] == ("output" if save_output else "temp")
    assert preview["filename"].endswith(".mp4")
    assert result["ui"]["animated"] == (True,)


@pytest.mark.parametrize("codec", ["h264_nvenc","libx264","hevc_nvenc","libx265","av1_nvenc","libsvtav1"])
@pytest.mark.parametrize("quality_preset,expected", [("custom",31),("low",28),("medium",23),("high",18)])
def test_h264_quality_presets_reach_encoder_and_keep_speed(tmp_path,monkeypatch,codec,quality_preset,expected):
    folders = ModuleType("folder_paths")
    folders.get_output_directory = lambda:str(tmp_path)
    folders.get_save_image_path = lambda *args:(str(tmp_path),"mAI",1,"",None)
    monkeypatch.setitem(sys.modules,"folder_paths",folders)
    calls = []
    def encode(frames,path,options,*args,**kwargs):
        calls.append(options)
        return 1,1/24
    monkeypatch.setattr(node_module,"encode_video",encode)
    MAIGPUVideoCombine().run(torch.zeros(1,4,4,3),codec=codec,quality=31,preset="p6",
                            h264_quality_preset=quality_preset)
    assert calls[0].quality == (expected if codec in ("h264_nvenc","libx264") else 31)
    assert calls[0].preset == "p6"
