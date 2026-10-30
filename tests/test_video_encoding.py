from dataclasses import replace
import io
import json
import os
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from utils.video_encoding import (
    NVENC_CODECS, SOFTWARE_CODECS, VideoEncodeOptions, _metadata_text, _pack_nv12, _write_all,
    _write_audio, build_encode_command, calculate_video_timing,
    check_nvenc_support, check_software_support, encode_video, find_ffmpeg, get_video_dimensions,
    playback_frame_count, playback_indices, resolve_h264_quality, validate_audio,
    validate_filename_prefix, validate_options,
)


@pytest.mark.parametrize("codec", ["h264_nvenc","libx264"])
@pytest.mark.parametrize("preset,expected", [("custom",31),("low",28),("medium",23),("high",18)])
def test_h264_quality_shortcuts_use_cq_or_crf_and_preserve_speed(codec,preset,expected):
    quality = resolve_h264_quality(codec,31,preset)
    options = VideoEncodeOptions(codec=codec,quality=quality,preset="p6")
    command = build_encode_command("ffmpeg","video.mp4",320,240,7,options)
    quality_flag = "-cq" if codec == "h264_nvenc" else "-crf"
    assert command[command.index(quality_flag)+1] == str(expected)
    assert command[command.index("-preset")+1] == ("p6" if codec == "h264_nvenc" else "slower")


def test_h264_quality_shortcuts_keep_legacy_default_and_reject_unknown_preset():
    assert resolve_h264_quality("h264_nvenc",31) == 31
    with pytest.raises(ValueError,match="Unknown H.264 quality preset"):
        resolve_h264_quality("h264_nvenc",23,"ultra")


@pytest.mark.parametrize("count,pingpong,loops,expected", [(1,True,2,3),(2,True,0,2),(4,False,1,8),(4,True,0,6),(4,True,2,18)])
def test_playback_count(count,pingpong,loops,expected):
    assert playback_frame_count(count,pingpong,loops) == expected


def test_pingpong_sequence_has_no_duplicated_endpoints_and_chunks_match():
    expected = [0,1,2,3,2,1,0,1,2,3,2,1]
    assert playback_indices(0,12,4,True).tolist() == expected
    parts = torch.cat([playback_indices(0,5,4,True),playback_indices(5,9,4,True),playback_indices(9,12,4,True)])
    assert parts.tolist() == expected
    assert playback_indices(0,8,4,False).tolist() == [0,1,2,3,0,1,2,3]


@pytest.mark.parametrize("channels", [1,3,4])
@pytest.mark.parametrize("dtype", [torch.float32,torch.float16,torch.bfloat16])
def test_chunk_packing_channels_quantization_padding_and_input_preservation(channels,dtype):
    frames = torch.full((2,3,5,channels),0.5,dtype=dtype)
    before = frames.clone()
    n,w,h = get_video_dimensions(frames)
    packed = _pack_nv12(frames,w,h)
    assert (n,w,h) == (2,6,4)
    assert packed.shape == (2,36) and packed.dtype == torch.uint8
    y = packed[:,:24].reshape(2,4,6)
    assert y[:,:3,:5].eq(126).all()
    assert y[:,3].eq(16).all() and y[:,:,5].eq(16).all()
    assert packed[:,24:].eq(128).all()
    torch.testing.assert_close(frames,before)


@pytest.mark.parametrize("frames", [None,torch.zeros(0,4,4,3),torch.zeros(1,4,0,3),torch.zeros(4,4,3),torch.zeros(1,4,4,2),torch.zeros(1,4,4,3,dtype=torch.uint8)])
def test_invalid_frames(frames):
    with pytest.raises(ValueError):
        get_video_dimensions(frames)


def test_nonfinite_frames_rejected_and_out_of_range_pixels_clipped():
    with pytest.raises(ValueError,match="non-finite"):
        _pack_nv12(torch.full((1,4,4,3),float("nan")),4,4)
    image = torch.tensor([-1.0,0.5,2.0]).view(1,1,1,3)
    torch.testing.assert_close(_pack_nv12(image,2,2),_pack_nv12(image.clamp(0,1),2,2))


@pytest.mark.parametrize("color,yuv", [
    ((0,0,0),(16,128,128)), ((1,1,1),(235,128,128)),
    ((1,0,0),(63,102,240)), ((0,1,0),(173,42,26)),
    ((0,0,1),(32,240,118)), ((0.5,0.5,0.5),(126,128,128)),
])
def test_bt709_limited_range_reference_colors(color,yuv):
    image = torch.tensor(color,dtype=torch.float32).view(1,1,1,3).expand(1,2,2,3)
    packed = _pack_nv12(image,2,2)[0]
    assert packed[:4].eq(yuv[0]).all()
    assert packed[4:].tolist() == list(yuv[1:])


@pytest.mark.parametrize("prefix", ["video/mAI","mAI","nested\\video","frames_%width%x%height%"])
def test_relative_prefixes(prefix):
    assert validate_filename_prefix(prefix) == prefix.replace("\\","/")


@pytest.mark.parametrize("prefix", [""," ",None,"../video","sub/../../video","/tmp/video","C:\\video","C:video","\\\\server\\video","folder/","video\nname","video?name"])
def test_unsafe_prefixes(prefix):
    with pytest.raises(ValueError):
        validate_filename_prefix(prefix)


@pytest.mark.parametrize("changes", [{"frame_rate":0},{"frame_rate":float("nan")},{"frame_rate":True},{"frame_rate":1001},{"codec":"unknown"},{"preset":"slow"},{"quality":-1},{"quality":52},{"gpu_device":-2},{"chunk_size":-1},{"loop_count":101},{"loop_count":0.5},{"pingpong":1},{"trim_to_audio":"false"}])
def test_invalid_options(changes):
    with pytest.raises(ValueError):
        validate_options(replace(VideoEncodeOptions(),**changes))


@pytest.mark.parametrize("audio", [{},{"waveform":torch.zeros(2,1,100),"sample_rate":48000},{"waveform":torch.zeros(1,0,100),"sample_rate":48000},{"waveform":torch.zeros(1,1,0),"sample_rate":48000},{"waveform":torch.zeros(1,1,100),"sample_rate":0}])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        validate_audio(audio)


def test_audio_timing_and_pcm_interleave(tmp_path):
    waveform = torch.tensor([[[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]])
    info = validate_audio({"waveform":waveform,"sample_rate":10})
    assert validate_audio(None) is None
    options = VideoEncodeOptions(frame_rate=10,pingpong=True,loop_count=1)
    assert calculate_video_timing(5,options,info) == (16,1.6)
    assert calculate_video_timing(5,replace(options,trim_to_audio=True),info) == (4,0.4)
    path = tmp_path/"audio.raw"
    _write_audio(path,info,0.2,None)
    data = torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.float32)
    torch.testing.assert_close(data,torch.tensor([0.1,0.5,0.2,0.6]))
    assert waveform[0,0,-1] == 0.4


@pytest.mark.parametrize("codec", NVENC_CODECS)
def test_command_uses_nvenc_and_fixed_argument_list(codec):
    options = VideoEncodeOptions(codec=codec,frame_rate=23.976,quality=19,preset="p6")
    info = (torch.zeros(1,2,48000),48000)
    command = build_encode_command("ffmpeg","some video.mp4",1920,1080,121,options,
                                   "audio.raw",info,"workflow.ffmeta")
    assert command[command.index("-c:v")+1] == codec
    assert command[command.index("-pixel_format")+1] == "nv12"
    assert command[command.index("-map_metadata")+1] == "2"
    assert command[command.index("-cq")+1] == "19"
    assert command[-1] == "some video.mp4"
    assert "-hwaccel" not in command and "-vf" not in command
    assert "-c:a" in command and "apad" in command
    silent = build_encode_command("ffmpeg","video.mp4",320,240,1,options)
    assert "-an" in silent and "-c:a" not in silent
    assert silent[silent.index("-map_metadata")+1] == "-1"


@pytest.mark.parametrize("codec", SOFTWARE_CODECS)
@pytest.mark.parametrize("preset,expected,svt", [("p1","ultrafast","12"),("p4","medium","6"),("p7","veryslow","3")])
def test_software_command_preserves_format_and_never_passes_nvenc_flags(codec,preset,expected,svt):
    options = VideoEncodeOptions(codec=codec,preset=preset,quality=19,gpu_device=128)
    validate_options(options)
    command = build_encode_command("ffmpeg","video.mp4",320,240,7,options)
    assert command[command.index("-c:v")+1] == codec
    assert command[command.index("-preset")+1] == (svt if codec == "libsvtav1" else expected)
    assert command[command.index("-crf")+1] == "19"
    assert command[command.index("-pixel_format")+1] == "nv12"
    assert command[command.index("-pix_fmt")+1] == "yuv420p"
    assert not set(("-gpu","-cq","-rc","-tune")) & set(command)
    assert command[-1] == "video.mp4"


@pytest.fixture
def ffmpeg_environment(monkeypatch):
    monkeypatch.delenv("MAI_FFMPEG_EXE",raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE",raising=False)
    module = ModuleType("imageio_ffmpeg")
    monkeypatch.setitem(sys.modules,"imageio_ffmpeg",module)
    return module


@pytest.mark.parametrize("variable", ["MAI_FFMPEG_EXE","IMAGEIO_FFMPEG_EXE"])
def test_explicit_ffmpeg_override_has_priority_over_path(monkeypatch,ffmpeg_environment,variable):
    monkeypatch.setenv(variable,"custom ffmpeg")
    monkeypatch.setattr(shutil,"which",lambda name: "chosen" if name == "custom ffmpeg" else "system")
    assert find_ffmpeg() == "chosen"


def test_mai_override_takes_priority_over_imageio_override(monkeypatch,ffmpeg_environment):
    monkeypatch.setenv("MAI_FFMPEG_EXE","mai")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE","imageio")
    monkeypatch.setattr(shutil,"which",lambda name:name)
    assert find_ffmpeg() == "mai"


def test_invalid_override_fails_instead_of_using_another_binary(monkeypatch,ffmpeg_environment):
    monkeypatch.setenv("MAI_FFMPEG_EXE","missing")
    monkeypatch.setattr(shutil,"which",lambda name:"system" if name == "ffmpeg" else None)
    with pytest.raises(RuntimeError,match="MAI_FFMPEG_EXE.*executable"):
        find_ffmpeg()


def test_system_ffmpeg_and_bundled_fallback(monkeypatch,ffmpeg_environment):
    monkeypatch.setattr(shutil,"which",lambda name:"system")
    assert find_ffmpeg() == "system"
    ffmpeg_environment.get_ffmpeg_exe = lambda:"bundled"
    monkeypatch.setattr(shutil,"which",lambda name:"bundled" if name == "bundled" else None)
    assert find_ffmpeg() == "bundled"


@pytest.mark.parametrize("failure", ["missing_module","missing_binary","imageio_error"])
def test_missing_ffmpeg_reports_cloud_installation_and_hardware_limit(monkeypatch,ffmpeg_environment,failure):
    monkeypatch.setattr(shutil,"which",lambda name:None)
    if failure == "missing_module":
        monkeypatch.setitem(sys.modules,"imageio_ffmpeg",None)
    elif failure == "missing_binary":
        ffmpeg_environment.get_ffmpeg_exe = lambda:"absent"
    else:
        def missing():
            raise RuntimeError("No ffmpeg exe could be found")
        ffmpeg_environment.get_ffmpeg_exe = missing
    with pytest.raises(RuntimeError) as error:
        find_ffmpeg()
    message = str(error.value)
    for expected in ("FFmpeg was not found", "pip install", "apt-get", "Modal", "MAI_FFMPEG_EXE", "B200/B300", "libx264"):
        assert expected in message


def test_software_capability_check_does_not_accept_missing_encoder(monkeypatch):
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout="",stderr="Codec not found"))
    with pytest.raises(RuntimeError,match="software encoder"):
        check_software_support("ffmpeg","libx265")


def test_software_encoding_does_not_request_nvenc_memory_or_test_nvenc(tmp_path,monkeypatch):
    import utils.video_encoding as encoding
    def unexpected(*args):
        pytest.fail("Software encoding must not request an NVENC context or capability check")
    monkeypatch.setattr(encoding,"video_memory_requirements",unexpected)
    monkeypatch.setattr(encoding,"check_nvenc_support",unexpected)
    monkeypatch.setattr(encoding,"find_ffmpeg",lambda:"ffmpeg")
    checks = []
    monkeypatch.setattr(encoding,"check_software_support",lambda *args:checks.append(args))
    class SuccessfulEncoder:
        def __init__(self,command,**kwargs):
            self.stdin = io.BytesIO()
            self.returncode = 0
            from pathlib import Path
            Path(command[-1]).write_bytes(b"video")
        def poll(self):
            return self.returncode
    monkeypatch.setattr(subprocess,"Popen",SuccessfulEncoder)
    output = tmp_path/"software.mp4"
    assert encode_video(torch.zeros(2,4,4,3),output,VideoEncodeOptions(codec="libx264")) == (2,2/24)
    assert checks == [("ffmpeg","libx264")]
    assert output.read_bytes() == b"video" and not list(tmp_path.glob(".mai_video_*"))


def test_metadata_is_in_a_file_and_escapes_reserved_characters():
    text = _metadata_text({"workflow":{"label":"equals=hash#semi;slash\\ newline\n"},"prompt":{"1":{}},"ignored":"no"})
    assert text.startswith(";FFMETADATA1\n")
    assert "workflow=" in text and "prompt=" in text and "ignored" not in text
    assert "\\=" in text and "\\#" in text and "\\;" in text


def test_partial_writes_are_completed():
    class PartialWriter(io.BytesIO):
        def write(self,data):
            return super().write(data[:3])
    stream = PartialWriter()
    _write_all(stream,b"1234567890")
    assert stream.getvalue() == b"1234567890"


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path/"existing.mp4"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        encode_video(torch.zeros(1,240,320,3),output)
    assert output.read_bytes() == b"keep"


def test_missing_nvenc_is_clear(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout="",stderr="Codec not found"))
    with pytest.raises(RuntimeError,match="matching software encoder"):
        check_nvenc_support("ffmpeg","h264_nvenc")


@pytest.mark.parametrize("failures", [1,2])
def test_cuda_oom_retries_once_after_scope_cleanup_with_same_codec(tmp_path,monkeypatch,failures):
    from contextlib import contextmanager
    import utils.video_encoding as encoding
    events = []
    @contextmanager
    def scope(device,required,force_offload=False):
        events.append(("enter",force_offload))
        try:
            yield
        finally:
            events.append(("exit",force_offload))
    monkeypatch.setattr(encoding,"gpu_memory_scope",scope)
    monkeypatch.setattr(encoding,"video_memory_requirements",lambda *args: {torch.device("cuda:1"): 100})
    calls = []
    def encode(frames,path,options,*args,**kwargs):
        calls.append(options)
        assert kwargs.get("cpu_conversion", False) == (options.codec in SOFTWARE_CODECS)
        if len(calls) <= failures:
            raise encoding.NVENCOutOfMemoryError("CUDA_ERROR_OUT_OF_MEMORY")
        return 3,3/24
    monkeypatch.setattr(encoding,"_encode_video",encode)
    options = VideoEncodeOptions(codec="hevc_nvenc",chunk_size=3)
    frames = torch.ones(3,4,4,3)
    assert encode_video(frames,tmp_path/"retry.mp4",options) == (3,3/24)
    assert len(calls) == failures + 1 and calls[0] == options
    assert calls[1] == replace(options,chunk_size=1)
    if failures == 2:
        assert calls[2] == replace(options,codec="libx265",chunk_size=1)
    assert events == [("enter",False),("exit",False),("enter",True),("exit",True)]
    assert frames.eq(1).all()


@pytest.mark.parametrize("error", [RuntimeError("codec unavailable"),InterruptedError("cancelled")])
def test_non_memory_errors_and_cancellation_never_retry(tmp_path,monkeypatch,error):
    import utils.video_encoding as encoding
    calls = []
    def encode(*args):
        calls.append(1)
        raise error
    monkeypatch.setattr(encoding,"_encode_video",encode)
    with pytest.raises(type(error),match=str(error)):
        encode_video(torch.zeros(1,4,4,3),tmp_path/"error.mp4")
    assert len(calls) == 1


def test_real_ffmpeg_cuda_context_error_is_classified_and_cleaned(tmp_path,monkeypatch):
    import utils.video_encoding as encoding
    monkeypatch.setattr(encoding,"find_ffmpeg",lambda:"ffmpeg")
    monkeypatch.setattr(encoding,"check_nvenc_support",lambda *args:None)
    calls = []
    class FailedEncoder:
        def __init__(self,*args,**kwargs):
            self.stdin = io.BytesIO()
            self.returncode = 1
            kwargs["stderr"].write(b"cuCtxCreate failed -> CUDA_ERROR_OUT_OF_MEMORY: out of memory\nNo capable devices found")
            calls.append(self)
        def poll(self):
            return self.returncode
    monkeypatch.setattr(subprocess,"Popen",FailedEncoder)
    with pytest.raises(encoding.NVENCOutOfMemoryError,match="CUDA_ERROR_OUT_OF_MEMORY"):
        encoding._encode_video(torch.zeros(2,4,4,3),tmp_path/"failure.mp4")
    assert len(calls) == 1 and all(call.stdin.closed for call in calls)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("codec,software", [("h264_nvenc","libx264"),("hevc_nvenc","libx265"),("av1_nvenc","libsvtav1")])
@pytest.mark.parametrize("error_type", ["nvenc", "cuda_oom"])
def test_hardware_failure_falls_back_once_with_matching_codec_and_all_inputs(tmp_path,monkeypatch,caplog,codec,software,error_type):
    import utils.video_encoding as encoding
    frames = torch.zeros(3,4,4,3)
    audio = {"waveform":torch.zeros(1,1,4800),"sample_rate":48000}
    metadata = {"workflow":{"nodes":[]}}
    options = VideoEncodeOptions(codec=codec,quality=19,preset="p3",pingpong=True,loop_count=1)
    calls = []
    def encode(batch,path,settings,sound,tags,interrupt,progress,**kwargs):
        calls.append(settings.codec)
        assert batch is frames and sound is audio and tags is metadata
        assert path == tmp_path/"fallback.mp4"
        if settings.codec == codec:
            error = encoding.NVENCError if error_type == "nvenc" else torch.cuda.OutOfMemoryError
            raise error("hardware unavailable")
        assert settings == replace(options,codec=software)
        assert kwargs == {"cpu_conversion":True}
        return 8,8/24
    monkeypatch.setattr(encoding,"_encode_video",encode)
    assert encode_video(frames,tmp_path/"fallback.mp4",options,audio,metadata) == (8,8/24)
    assert calls == [codec,software]
    assert "hardware unavailable" in caplog.text and software in caplog.text


@pytest.mark.parametrize("device", ["cpu","cuda"])
@pytest.mark.parametrize("pingpong", [False,True])
@pytest.mark.parametrize("count", [1,2,5])
def test_cpu_playback_chunks_preserve_sequence_across_turns_and_repeats(device,pingpong,count):
    from utils.video_encoding import _cpu_playback_chunk
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    frames = torch.arange(count,device=device)[:,None,None,None].expand(count,2,2,3)
    before = frames.clone()
    for start,end in ((0,1),(0,27),(3,17),(5,8)):
        result = _cpu_playback_chunk(frames,start,end,pingpong)
        assert result.device.type == "cpu"
        expected = frames.cpu().index_select(0,playback_indices(start,end,count,pingpong))
        torch.testing.assert_close(result,expected)
    torch.testing.assert_close(frames,before)


def test_failed_cpu_fallback_reports_both_errors_and_cleans_partial_files(tmp_path,monkeypatch):
    import utils.video_encoding as encoding
    monkeypatch.setattr(encoding,"find_ffmpeg",lambda:"ffmpeg")
    monkeypatch.setattr(encoding,"check_nvenc_support",lambda *args:None)
    monkeypatch.setattr(encoding,"check_software_support",lambda *args:None)
    codecs = []
    class FailedEncoder:
        def __init__(self,command,**kwargs):
            from pathlib import Path
            self.stdin = io.BytesIO()
            self.returncode = 1
            codec = command[command.index("-c:v")+1]
            codecs.append(codec)
            Path(command[-1]).write_bytes(b"partial")
            kwargs["stderr"].write(b"No capable devices found" if codec in NVENC_CODECS else b"software encoder failed")
        def poll(self):
            return self.returncode
    monkeypatch.setattr(subprocess,"Popen",FailedEncoder)
    with pytest.raises(RuntimeError) as error:
        encode_video(torch.zeros(2,4,4,3),tmp_path/"failure.mp4")
    assert "No capable devices found" in str(error.value)
    assert "CPU fallback (libx264) also failed" in str(error.value)
    assert "software encoder failed" in str(error.value)
    assert codecs == ["h264_nvenc","libx264"]
    assert list(tmp_path.iterdir()) == []


def test_cancellation_after_hardware_failure_does_not_start_cpu_fallback(tmp_path,monkeypatch):
    import utils.video_encoding as encoding
    calls = []
    def encode(*args):
        calls.append("nvenc")
        raise encoding.NVENCError("no device")
    def interrupt():
        if calls:
            raise InterruptedError("cancelled")
    monkeypatch.setattr(encoding,"_encode_video",encode)
    with pytest.raises(InterruptedError,match="cancelled"):
        encode_video(torch.zeros(1,4,4,3),tmp_path/"cancelled.mp4",check_interrupt=interrupt)
    assert calls == ["nvenc"]


NVENC_TESTS = os.environ.get("MAI_TEST_NVENC") == "1" and shutil.which("ffmpeg") and shutil.which("ffprobe")
nvenc = pytest.mark.skipif(not NVENC_TESTS,reason="Set MAI_TEST_NVENC=1 with FFmpeg/ffprobe and an NVENC-capable GPU")


def _probe(path):
    result = subprocess.run([shutil.which("ffprobe"),"-v","error","-show_streams","-show_format","-of","json",str(path)],capture_output=True,text=True,encoding="utf-8",check=True)
    return json.loads(result.stdout)


def _decode(path,width,height):
    result = subprocess.run([shutil.which("ffmpeg"),"-v","error","-i",str(path),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"],capture_output=True,check=True)
    return torch.frombuffer(bytearray(result.stdout),dtype=torch.uint8).reshape(-1,height,width,3).float()/255


ffmpeg = pytest.mark.skipif(
    os.environ.get("MAI_TEST_FFMPEG") != "1" or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Set MAI_TEST_FFMPEG=1 with FFmpeg/ffprobe for software integration tests",
)


@ffmpeg
@pytest.mark.parametrize("device", ["cpu","cuda"])
@pytest.mark.parametrize("failure", ["missing_encoder","driver_failure"])
def test_real_automatic_cpu_fallback_with_audio_metadata_and_sequence(tmp_path,monkeypatch,device,failure):
    import utils.video_encoding as encoding
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    frames = torch.linspace(0,1,5,device=device)[:,None,None,None].expand(5,25,31,3)
    before = frames.clone()
    metadata = {"workflow":{"label":"CPU fallback 雪"},"prompt":{"1":{}}}
    audio = {"waveform":torch.zeros(1,1,4800),"sample_rate":48000}
    actual_run = subprocess.run
    if failure == "missing_encoder":
        def run(command,**kwargs):
            if "encoder=h264_nvenc" in command:
                return SimpleNamespace(returncode=0,stdout="",stderr="Codec not found")
            return actual_run(command,**kwargs)
        monkeypatch.setattr(subprocess,"run",run)
    actual_pack = encoding._pack_nv12
    packed_devices = []
    def pack(chunk,width,height):
        packed_devices.append(chunk.device.type)
        return actual_pack(chunk,width,height)
    monkeypatch.setattr(encoding,"_pack_nv12",pack)
    progress = []
    output = tmp_path/"fallback.mp4"
    options = VideoEncodeOptions(gpu_device=128,chunk_size=3,pingpong=True,loop_count=1)
    assert encode_video(frames,output,options,audio,metadata,on_progress=lambda *p:progress.append(p)) == (16,16/24)
    info = _probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "h264" and (video["width"],video["height"]) == (32,26)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert json.loads(info["format"]["tags"]["workflow"]) == metadata["workflow"]
    decoded = _decode(output,32,26)
    expected = torch.tensor([0,.25,.5,.75,1,.75,.5,.25]*2)
    torch.testing.assert_close(decoded[:,:24,:30].mean((1,2,3)),expected,atol=.04,rtol=0)
    assert packed_devices[-6:] == ["cpu"]*6 and progress[-1] == (16,16)
    torch.testing.assert_close(frames,before)
    assert not list(tmp_path.glob(".mai_video_*"))


@ffmpeg
@pytest.mark.parametrize("codec", SOFTWARE_CODECS)
@pytest.mark.parametrize("device", ["cpu","cuda"])
def test_real_software_audio_metadata_dimensions_and_playback_without_nvenc(tmp_path,codec,device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA frame conversion needs a CUDA-capable PyTorch environment")
    frames = torch.zeros(7,241,321,3,device=device)
    frames[:3,...,0] = 1
    frames[3:,...,2] = 1
    before = frames.clone()
    audio = {"waveform":torch.zeros(1,2,4800),"sample_rate":48000}
    metadata = {"workflow":{"label":"a=b;#雪"},"prompt":{"1":{}}}
    output = tmp_path/f"{codec}.mp4"
    options = VideoEncodeOptions(codec=codec,preset="p3",chunk_size=3,pingpong=True,loop_count=1)
    count,duration = encode_video(frames,output,options,audio,metadata)
    assert (count,duration) == (24,1.0)
    info = _probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    sound = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert video["codec_name"] == {"libx264":"h264","libx265":"hevc","libsvtav1":"av1"}[codec]
    assert (video["width"],video["height"],int(video["nb_frames"])) == (322,242,24)
    assert video["pix_fmt"] == "yuv420p"
    assert video["color_space"] == "bt709" and video["color_range"] == "tv"
    assert sound["codec_name"] == "aac" and sound["channels"] == 2
    assert abs(float(sound["duration"])-duration) < 0.05
    assert json.loads(info["format"]["tags"]["workflow"]) == metadata["workflow"]
    decoded = _decode(output,322,242)
    indices = playback_indices(0,count,7,True)
    reds = indices < 3
    assert decoded[reds,:238,:318,0].mean() > 0.94
    assert decoded[~reds,:238,:318,2].mean() > 0.94
    torch.testing.assert_close(frames,before)
    assert not list(tmp_path.glob(".mai_video_*"))


@nvenc
@pytest.mark.parametrize("codec", NVENC_CODECS)
def test_real_nvenc_audio_metadata_dimensions_and_order(tmp_path,codec):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = torch.zeros(7,241,321,3,device=device)
    frames[:3,...,0] = 1
    frames[3:,...,2] = 1
    before = frames.clone()
    audio = {"waveform":torch.sin(torch.arange(4800,device=device)*0.0576)[None,None,:]*0.2,"sample_rate":48000}
    output = tmp_path/f"{codec}.mp4"
    metadata = {"workflow":{"text":"a=b;#slash\\\n雪","large":"x"*40000},"prompt":{"1":{}}}
    count,duration = encode_video(frames,output,VideoEncodeOptions(codec=codec,chunk_size=3),audio,metadata)
    assert count == 7 and duration == 7/24
    info = _probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    sound = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert video["codec_name"] == {"h264_nvenc":"h264","hevc_nvenc":"hevc","av1_nvenc":"av1"}[codec]
    assert (video["width"],video["height"],int(video["nb_frames"])) == (322,242,7)
    assert video["pix_fmt"] == "yuv420p"
    assert video["color_space"] == "bt709" and video["color_range"] == "tv"
    assert sound["codec_name"] == "aac" and sound["channels"] == 1
    assert abs(float(sound["duration"])-duration) < 0.05
    assert json.loads(info["format"]["tags"]["workflow"]) == metadata["workflow"]
    assert json.loads(info["format"]["tags"]["prompt"]) == metadata["prompt"]
    decoded = _decode(output,322,242)
    assert decoded.shape[0] == 7
    assert decoded[0,:238,:318,0].mean() > 0.96
    assert decoded[-1,:238,:318,2].mean() > 0.96
    torch.testing.assert_close(frames,before)
    assert not list(tmp_path.glob(".mai_video_*"))


@nvenc
def test_real_pingpong_repeats_silent_video_and_121_frame_batch(tmp_path):
    frames = torch.linspace(0,1,6)[:,None,None,None].expand(6,240,320,3)
    output = tmp_path/"pingpong.mp4"
    count,duration = encode_video(frames,output,VideoEncodeOptions(pingpong=True,loop_count=1,chunk_size=3))
    assert (count,duration) == (20,20/24)
    decoded = _decode(output,320,240)
    expected = torch.tensor([0,.2,.4,.6,.8,1,.8,.6,.4,.2]*2)
    torch.testing.assert_close(decoded.mean((1,2,3)),expected,atol=0.03,rtol=0)
    assert len(_probe(output)["streams"]) == 1
    batch = torch.linspace(0,1,121)[:,None,None,None].expand(121,240,320,3)
    path = tmp_path/"121.mp4"
    assert encode_video(batch,path)[0] == 121
    video = _probe(path)["streams"][0]
    assert int(video["nb_frames"]) == 121
    decoded = _decode(path,320,240)
    assert decoded[0].mean() < 0.02 and decoded[-1].mean() > 0.97


@nvenc
@pytest.mark.parametrize("audio_samples,trim,expected", [(7200,False,24),(7200,True,4),(96000,False,24)])
def test_real_audio_padding_trimming_and_duration(tmp_path,audio_samples,trim,expected):
    audio = {"waveform":torch.zeros(1,2,audio_samples),"sample_rate":48000}
    output = tmp_path/"timing.mp4"
    count,duration = encode_video(torch.zeros(24,240,320,3),output,
                                   VideoEncodeOptions(trim_to_audio=trim),audio)
    assert count == expected
    info = _probe(output)
    sound = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert sound["channels"] == 2 and abs(float(sound["duration"])-duration) < 0.05


@nvenc
def test_real_failure_and_interrupt_cleanup(tmp_path):
    destination = tmp_path/"failed.mp4"
    assert encode_video(torch.zeros(2,240,320,3),destination,VideoEncodeOptions(gpu_device=128)) == (2,2/24)
    assert _probe(destination)["streams"][0]["codec_name"] == "h264"
    assert not list(tmp_path.glob(".mai_video_*"))
    destination.unlink()
    frames = torch.zeros(16,240,320,3)
    def interrupt():
        raise InterruptedError("cancelled")
    def progress(current,total):
        interrupt()
    with pytest.raises(InterruptedError,match="cancelled"):
        encode_video(frames,destination,VideoEncodeOptions(chunk_size=2),on_progress=progress)
    assert not destination.exists() and not list(tmp_path.glob(".mai_video_*"))
