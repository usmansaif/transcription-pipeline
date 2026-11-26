import argparse
import json
from pathlib import Path


def load_model(size: str = "base"):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, wav: Path) -> dict:
    segments, info = model.transcribe(str(wav), vad_filter=True)
    return {
        "language": info.language,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in segments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a 16 kHz WAV into raw Whisper segments.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    raw = transcribe(load_model(args.model), args.wav)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
