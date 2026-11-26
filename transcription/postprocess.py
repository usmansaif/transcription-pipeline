import argparse
import json
import re
from pathlib import Path

FILLERS = re.compile(r"\b(?:um+|uh+|erm?|hmm+)\b[,.]?\s*", re.IGNORECASE)
REPEATS = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = REPEATS.sub(r"\1", FILLERS.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def postprocess(data: dict) -> dict:
    segments = []
    for segment in data["segments"]:
        text = clean_text(segment["text"])
        if text:
            segments.append({**segment, "text": text})
    return {
        "language": data.get("language"),
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }


def save(result: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"[{format_time(s['start'])} - {format_time(s['end'])}] {s['text']}"
             for s in result["segments"]]
    (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean segments and write JSON + TXT.")
    parser.add_argument("segments", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()

    data = json.loads(args.segments.read_text(encoding="utf-8"))
    save(postprocess(data), args.out_dir, args.name)


if __name__ == "__main__":
    main()
