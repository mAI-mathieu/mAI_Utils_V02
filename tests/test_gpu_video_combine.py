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
    assert inputs["optional"] == {"audio":("AUDIO",)}
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
    assert calls[0][3] == ({"prompt":{"test":{}},"workflow":workflow} if save_metadata else {})
    preview = result["ui"]["images"][0]
    assert preview["type"] == ("output" if save_output else "temp")
    assert preview["filename"].endswith(".mp4")
    assert result["ui"]["animated"] == (True,)
