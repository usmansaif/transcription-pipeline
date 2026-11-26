import argparse
import logging
import tempfile
from pathlib import Path

from transcription import transcribe as stt
from transcription.build_segments import build_segments
from transcription.chunking import split_wav
from transcription.load_convert import load_convert
from transcription.postprocess import postprocess, save

log = logging.getLogger(__name__)


def run(audio: Path, out_dir: Path, model_size: str = "base", chunk_seconds: float = 600.0) -> dict:
    audio = Path(audio)
    with tempfile.TemporaryDirectory(prefix="transcription_") as tmp:
        work = Path(tmp)
        wav = load_convert(audio, work / "audio.wav")
        chunks = split_wav(wav, work, chunk_seconds)

        model = stt.load_model(model_size)
        language, raw_segments = None, []
        for number, (chunk, offset) in enumerate(chunks, 1):
            log.info("transcribing chunk %d/%d", number, len(chunks))
            raw = stt.transcribe(model, chunk)
            language = language or raw["language"]
            for segment in raw["segments"]:
                segment["start"] += offset
                segment["end"] += offset
            raw_segments.extend(raw["segments"])

    result = postprocess(build_segments({"language": language, "segments": raw_segments}))
    save(result, Path(out_dir), audio.stem)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a WAV/MP3 file into JSON and TXT.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--model", default="base")
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(args.audio, args.out_dir, args.model, args.chunk_seconds)
    log.info("%d segments written to %s", len(result["segments"]), args.out_dir)


if __name__ == "__main__":
    main()
