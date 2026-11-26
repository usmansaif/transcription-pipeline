import json
import math
import shutil
import struct
import subprocess
import wave

import pytest

from transcription import transcribe as stt
from transcription.build_segments import build_segments
from transcription.chunking import split_wav
from transcription.load_convert import InvalidAudioError, load_convert
from transcription.pipeline import run
from transcription.postprocess import clean_text

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
RATE = 16_000


def write_wav(path, seconds, silence=None):
    samples = [int(8000 * math.sin(2 * math.pi * 440 * i / RATE)) for i in range(int(seconds * RATE))]
    if silence:
        samples[int(silence[0] * RATE):int(silence[1] * RATE)] = [0] * int((silence[1] - silence[0]) * RATE)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


@pytest.fixture
def fake_stt(monkeypatch):
    monkeypatch.setattr(stt, "load_model", lambda size: None)
    monkeypatch.setattr(stt, "transcribe", lambda model, wav: {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " um so thanks for for joining",
             "avg_logprob": -0.1, "no_speech_prob": 0.01},
            {"start": 2.0, "end": 3.0, "text": " uh", "avg_logprob": -0.5, "no_speech_prob": 0.1},
            {"start": 3.0, "end": 4.0, "text": " (music)", "avg_logprob": -2.0, "no_speech_prob": 0.9},
        ],
    })


def test_clean_text_removes_fillers_and_repeats():
    assert clean_text("um so thanks for for joining") == "So thanks for joining"
    assert clean_text("uh, um") == ""


def test_build_segments_filters_sorts_and_removes_overlap():
    raw = {"language": "en", "segments": [
        {"start": 2, "end": 6, "text": " b", "avg_logprob": -0.2, "no_speech_prob": 0.0},
        {"start": 0, "end": 3, "text": " a", "avg_logprob": -0.2, "no_speech_prob": 0.0},
        {"start": 6, "end": 7, "text": " noise", "avg_logprob": -3.0, "no_speech_prob": 0.9},
        {"start": 7, "end": 8, "text": "  ", "avg_logprob": -0.1, "no_speech_prob": 0.0},
    ]}
    segments = build_segments(raw)["segments"]
    assert [(s["text"], s["start"], s["end"]) for s in segments] == [("a", 0.0, 2.0), ("b", 2.0, 6.0)]
    assert 0 < segments[0]["confidence"] <= 1


def test_split_wav_cuts_inside_silence(tmp_path):
    wav = write_wav(tmp_path / "gap.wav", 25, silence=(11.0, 11.4))
    chunks = split_wav(wav, tmp_path, chunk_seconds=10)
    assert len(chunks) == 2
    assert 11.0 <= chunks[1][1] <= 11.4


def test_short_audio_is_not_split(tmp_path):
    wav = write_wav(tmp_path / "short.wav", 5)
    assert split_wav(wav, tmp_path, chunk_seconds=10) == [(wav, 0.0)]


@needs_ffmpeg
def test_load_convert_resamples_stereo(tmp_path):
    src = tmp_path / "in.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1:sample_rate=44100",
                    "-ac", "2", str(src)], check=True)
    out = load_convert(src, tmp_path / "out.wav")
    with wave.open(str(out), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels()) == (RATE, 1)


def test_load_convert_rejects_bad_input(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(InvalidAudioError):
        load_convert(txt, tmp_path / "o.wav")
    with pytest.raises(FileNotFoundError):
        load_convert(tmp_path / "missing.wav", tmp_path / "o.wav")


@needs_ffmpeg
def test_load_convert_rejects_corrupt_mp3(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(InvalidAudioError):
        load_convert(bad, tmp_path / "o.wav")


@needs_ffmpeg
def test_pipeline_end_to_end_with_mp3(tmp_path, fake_stt):
    wav = write_wav(tmp_path / "call.wav", 4)
    mp3 = tmp_path / "call.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True)

    result = run(mp3, tmp_path / "out")

    assert result["text"] == "So thanks for joining"
    saved = json.loads((tmp_path / "out" / "call.json").read_text())
    assert saved["segments"][0]["start"] == 0.0
    assert (tmp_path / "out" / "call.txt").read_text() == "[00:00 - 00:02] So thanks for joining\n"


@needs_ffmpeg
def test_pipeline_offsets_timestamps_across_chunks(tmp_path, fake_stt):
    wav = write_wav(tmp_path / "long.wav", 30, silence=(10.5, 10.9))
    result = run(wav, tmp_path / "out", chunk_seconds=10)
    starts = [s["start"] for s in result["segments"]]
    assert len(starts) >= 3
    assert starts == sorted(starts)
    assert starts[1] >= 10.5
