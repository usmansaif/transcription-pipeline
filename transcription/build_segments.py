import argparse
import json
import math
from pathlib import Path

MAX_NO_SPEECH_PROB = 0.6


def build_segments(raw: dict) -> dict:
    segments = []
    for item in raw["segments"]:
        text = item["text"].strip()
        if not text or item.get("no_speech_prob", 0.0) > MAX_NO_SPEECH_PROB:
            continue
        confidence = None
        if "avg_logprob" in item:
            confidence = round(min(1.0, max(0.0, math.exp(item["avg_logprob"]))), 3)
        segments.append({
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": text,
            "confidence": confidence,
        })

    segments.sort(key=lambda s: s["start"])
    for current, following in zip(segments, segments[1:]):
        current["end"] = min(current["end"], following["start"])
    for segment in segments:
        segment["end"] = max(segment["end"], segment["start"])
        segment["start"] = round(segment["start"], 2)
        segment["end"] = round(segment["end"], 2)

    return {"language": raw.get("language"), "segments": segments}


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn raw Whisper output into timed segments.")
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    result = build_segments(json.loads(args.raw.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
