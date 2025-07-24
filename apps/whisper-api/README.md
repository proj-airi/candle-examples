# whisper-api

> An OpenAI-compatible speech transcription API service with real-time streaming response (SSE), integrated with Silero VAD and Whisper models.
>
> This service provides a `/v1/audio/transcriptions` endpoint that is fully compatible with OpenAI's API format, supporting both standard JSON responses and streaming Server-Sent Events.

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/whisper-api
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package whisper-api --features cuda
```

### macOS Metal

```
cargo build --package whisper-api --features metal
```

### CPU Only

```
cargo build --package whisper-api
```

## Run

### Any platforms

```shell
cargo run --package whisper-api --release
```

The server will start at `http://localhost:3000`.

## Usage

### Basic transcription

```bash
curl -X POST http://localhost:3000/v1/audio/transcriptions \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1"
```

### Streaming transcription

```bash
curl -X POST "http://localhost:3000/v1/audio/transcriptions?stream=true" \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1" \
  --no-buffer
```

## API Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ | Audio file to transcribe |
| `model` | String | ❌ | Model name (default: "whisper-1") |
| `language` | String | ❌ | Audio language |
| `prompt` | String | ❌ | Prompt text |
| `response_format` | String | ❌ | Response format (default: "json") |
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0) |
| `stream` | Boolean | ❌ | Enable streaming response (query parameter) |

## Supported Audio Formats

- WAV, MP3, FLAC, M4A
- Any format supported by Symphonia

## Environment Variables

```bash
# Set log level
export RUST_LOG=debug

# Force CPU usage
export CANDLE_FORCE_CPU=1

# Disable VAD (Voice Activity Detection) - process entire audio directly
export DISABLE_VAD=true

# Set VAD threshold (0.0-1.0, default: 0.15)
export VAD_THRESHOLD=0.2
```

## Acknowledgements

- [candle](https://github.com/huggingface/candle) - High-performance ML framework
- [axum](https://github.com/tokio-rs/axum) - Modern web framework
- [OpenAI](https://openai.com/) - API design reference
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection model
