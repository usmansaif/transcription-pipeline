import wave
from array import array
from pathlib import Path

FRAME_SECONDS = 0.1


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def _quietest_frame(samples: array, frame_len: int) -> int:
    best_index, best_energy = 0, None
    for i in range(0, len(samples) - frame_len + 1, frame_len):
        energy = sum(abs(x) for x in samples[i:i + frame_len])
        if best_energy is None or energy < best_energy:
            best_index, best_energy = i, energy
    return best_index


def split_wav(path: Path, out_dir: Path, chunk_seconds: float = 600.0,
              search_seconds: float = 5.0) -> list[tuple[Path, float]]:
    """Split into ~chunk_seconds pieces, cutting at the quietest 100 ms near each boundary.

    Returns (chunk_path, start_offset_seconds) pairs.
    """
    with wave.open(str(path), "rb") as src:
        rate, total = src.getframerate(), src.getnframes()
        chunk_len = int(chunk_seconds * rate)
        search_len = int(search_seconds * rate)
        frame_len = int(FRAME_SECONDS * rate)

        if total <= chunk_len + search_len:
            return [(path, 0.0)]

        cuts = [0]
        while total - cuts[-1] > chunk_len + search_len:
            target = cuts[-1] + chunk_len
            start = max(cuts[-1] + frame_len, target - search_len)
            end = min(total, target + search_len)
            src.setpos(start)
            samples = array("h")
            samples.frombytes(src.readframes(end - start))
            cuts.append(start + _quietest_frame(samples, frame_len) + frame_len // 2)
        cuts.append(total)

        pieces = []
        for index, (begin, finish) in enumerate(zip(cuts, cuts[1:])):
            piece = out_dir / f"chunk_{index:04d}.wav"
            src.setpos(begin)
            with wave.open(str(piece), "wb") as dst:
                dst.setnchannels(1)
                dst.setsampwidth(2)
                dst.setframerate(rate)
                dst.writeframes(src.readframes(finish - begin))
            pieces.append((piece, begin / rate))
    return pieces
