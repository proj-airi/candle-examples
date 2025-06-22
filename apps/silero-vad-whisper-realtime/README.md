# silero-vad-whisper-realtime

Real-time speech transcription using Silero VAD (Voice Activity Detection) combined with Whisper for accurate speech-to-text conversion.

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/silero-vad-whisper-realtime
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package silero-vad-whisper-realtime --features cuda
```

### macOS

```
cargo build --package silero-vad-whisper-realtime --features metal
```

## Run

### NVIDIA CUDA

```shell
cargo run --package silero-vad-whisper-realtime --features cuda
```

### macOS

```shell
cargo run --package silero-vad-whisper-realtime --features metal
```

## Command Line Options

The application supports several configuration options:

- `--cpu`: Force CPU usage instead of GPU acceleration
- `--sample-rate <RATE>`: Audio sample rate (default: 16000)
- `--device <DEVICE>`: Specify audio input device
- `--vad-threshold <THRESHOLD>`: VAD sensitivity threshold (default: 0.3)
- `--min-silence-duration-ms <MS>`: Minimum silence duration to end speech segment (default: 500ms)
- `--min-speech-duration-ms <MS>`: Minimum speech duration to start transcription (default: 100ms)
- `--whisper-model <MODEL>`: Whisper model size (default: medium)

### Example with custom settings

```shell
cargo run --package silero-vad-whisper-realtime --features cuda -- --vad-threshold 0.5 --whisper-model large
```

## Features

- **Silero VAD Integration**: Intelligent voice activity detection to reduce unnecessary processing
- **Real-time Processing**: Continuous audio stream processing with minimal latency
- **Configurable Parameters**: Adjustable VAD thresholds and timing parameters
- **Multiple Whisper Models**: Support for different Whisper model sizes (tiny, small, medium, large)
- **Cross-platform**: Works on CUDA-enabled systems and macOS with Metal acceleration

## How It Works

The application combines two powerful technologies:

1. **Silero VAD**: Detects when speech is present in the audio stream, filtering out silence and background noise
2. **Whisper**: Transcribes the detected speech segments into text

This approach significantly improves efficiency by only running Whisper inference on audio segments that actually contain speech.

## Acknowledgements

- [candle/candle-examples/examples/whisper](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)
- [candle/candle-wasm-examples/whisper](https://github.com/huggingface/candle/tree/main/candle-wasm-examples/whisper)
