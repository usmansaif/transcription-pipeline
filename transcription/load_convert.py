import argparse
import shutil
import subprocess
from pathlib import Path

ALLOWED_SUFFIXES = {".wav", ".mp3"}
SAMPLE_RATE = 16_000


class InvalidAudioError(ValueError):
    pass


def load_convert(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.suffix.lower() not in ALLOWED_SUFFIXES:
        raise InvalidAudioError(f"unsupported format: {src.suffix or 'none'}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(src.resolve()),
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise InvalidAudioError(f"cannot decode {src.name}: {exc.stderr.strip()}") from exc
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WAV/MP3 to 16 kHz mono WAV.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(load_convert(args.audio, args.output))


if __name__ == "__main__":
    main()
