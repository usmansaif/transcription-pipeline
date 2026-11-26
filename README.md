# Transcription Pipeline

Turns a WAV or MP3 file into a cleaned transcript with per-segment timestamps, saved as JSON and TXT.
Open-source speech-to-text ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)), runs locally, no API key.

```
audio (WAV/MP3) -> load_convert -> [split_wav] -> transcribe -> build_segments -> postprocess -> JSON + TXT
```

## Usage

Requires Python 3.10+ and `ffmpeg` on the PATH.

```bash
pip install -r requirements.txt
python -m transcription.pipeline call.mp3 --out-dir out/ --model base
```

Every stage is also a standalone command that reads and writes files:

```bash
python -m transcription.load_convert call.mp3 work/audio.wav
python -m transcription.transcribe work/audio.wav work/raw.json --model base
python -m transcription.build_segments work/raw.json work/segments.json
python -m transcription.postprocess work/segments.json out/ call
```

Output (`out/call.json`):

```json
{
  "language": "en",
  "text": "So thanks for joining. We reviewed the roadmap.",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "So thanks for joining.", "confidence": 0.905},
    {"start": 3.2, "end": 6.0, "text": "We reviewed the roadmap.", "confidence": 0.741}
  ]
}
```

`out/call.txt` has one line per segment: `[00:00 - 00:03] So thanks for joining.`

Tests: `pip install -r requirements-dev.txt && python -m pytest` (the speech model is faked, so no download is needed).

## Design decisions

| Stage | Module | Decision |
|---|---|---|
| Load and convert | `load_convert.py` | ffmpeg converts everything to 16 kHz mono 16-bit WAV, the format Whisper uses internally. One code path for every input. Unsupported extensions and undecodable files raise `InvalidAudioError`, kept separate from system errors. |
| Chunking | `chunking.py` | Recordings over `--chunk-seconds` (default 600) are split at the quietest 100 ms near each boundary so words are not cut in half. Timestamps are shifted back onto the original timeline. Frames are streamed, so memory stays flat. |
| Transcribe | `transcribe.py` | faster-whisper with int8 on CPU and VAD filtering, which skips silence and reduces hallucinated text. The model is loaded once and reused for all chunks. Saves the raw output unchanged. |
| Segments | `build_segments.py` | Pure function. Drops non-speech and empty segments, converts log-probability to a 0-1 confidence, sorts, and removes overlaps. |
| Post-process | `postprocess.py` | Pure function. Removes fillers and repeated words, fixes spacing and capitalization, writes JSON and TXT. |

- **Raw output and filtering are separate stages.** Changing a threshold or the cleaning rules does not require re-running the slow transcription.
- **Stages 3 and 4 are pure functions**, tested without a model or audio.
- **Formats.** Adding a format means adding its extension; ffmpeg already decodes it.
- **Long files.** faster-whisper decodes in 30 s windows and streams segments. Chunking adds bounded memory and gives the unit of work for parallelism and retries (below).
- **Why not an API.** A local model keeps audio on the machine and has no per-minute cost. The trade-off is CPU time; swap `transcribe.py` for a hosted service if latency matters more.

## System design

```
client -> API -> object storage (audio) |-> queue -> workers (model loaded once per process) -> object storage / database (transcripts)
```

### Concurrent uploads

- The API is stateless and only accepts, stores, enqueues and returns `202 Accepted` with a job id. Transcription never runs in the request path.
- Large files upload straight to object storage through presigned URLs, so the API servers never carry the audio bytes.
- A queue (SQS, RabbitMQ or Redis) sits between the API and the workers. Workers are CPU-bound, so each runs one model instance and a concurrency limit matched to its cores. Autoscale on queue depth.
- Backpressure: a bounded queue and per-client rate limits that return `429`.
- A long file becomes several chunk jobs, so one large upload spreads across many workers.

### Storage

- **Audio:** object storage (S3, GCS or MinIO), keyed by content hash so duplicate uploads are stored once. A lifecycle rule deletes originals after a retention period; encrypted at rest.
- **Jobs and transcripts:** PostgreSQL.
  - `jobs`: id, status, audio_key, model, attempts, error, timestamps.
  - `transcripts`: job_id, language, text, segments (JSONB).
  - Very large transcripts go to object storage with a pointer in the row.
- Search across transcripts later means a full-text index on the text, or a segments table.

### Retries and recovery

- Job states: `queued -> processing -> done | failed`.
- Delivery is at-least-once, so processing is idempotent: results are keyed by job id and written with an upsert.
- Permanent errors (`InvalidAudioError`) fail immediately with the reason stored. Transient errors (worker crash, out of memory, storage timeout) retry with exponential backoff and jitter up to a maximum, then go to a dead-letter queue and raise an alert.
- Workers heartbeat while processing; when the queue's visibility timeout expires after a crash, the job returns to the queue.
- Chunk results are checkpointed, so a retry resumes at the failed chunk instead of restarting the file.
- Failed jobs can be re-queued manually.

### API

| Method and path | Purpose |
|---|---|
| `POST /v1/transcriptions` | Multipart upload or `{"audio_url": ...}`. Optional `Idempotency-Key` header and `callback_url`. Returns `202` with the job id. |
| `GET /v1/transcriptions/{id}` | Status, attempts, error; result included when done. |
| `GET /v1/transcriptions/{id}/transcript?format=json\|txt\|srt\|vtt` | The transcript in the requested format. |
| `GET /v1/transcriptions` | List with pagination and status filter. |
| `DELETE /v1/transcriptions/{id}` | Remove the audio and transcript. |

- Auth by API key; versioned path; OpenAPI spec generated (FastAPI).
- Errors: `415` unsupported type, `413` too large, `422` undecodable audio, `429` rate limited.
- Clients poll the status endpoint or receive a webhook. Very short clips can offer an optional synchronous mode.

## Known limits

- Fillers and repeated words are removed with regexes, so a legitimate "that that" is collapsed.
- Context is not carried across chunk boundaries.
- No speaker labels or word-level timestamps.