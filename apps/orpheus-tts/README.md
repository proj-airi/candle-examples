# orpheus-tts

High-quality text-to-speech synthesis using Orpheus TTS models with multiple voice options, powered by Rust and Candle framework.

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/orpheus-tts
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package orpheus-tts --features cuda
```

### macOS

```
cargo build --package orpheus-tts --features metal
```

## Run

### NVIDIA CUDA

```shell
cargo run --package orpheus-tts --features cuda -- --hf-token YOUR_HF_TOKEN
```

### macOS

```shell
cargo run --package orpheus-tts --features metal -- --hf-token YOUR_HF_TOKEN
```

## Command Line Options

The application supports several configuration options:

- `--cpu`: Force CPU usage instead of GPU acceleration
- `--hf-token <TOKEN>`: HuggingFace API token (required for model access)
- `--prompt <TEXT>`: Text to synthesize (default: "Hey, how are you doing today?")
- `--voice <VOICE>`: Voice to use (default: tara). Available voices: tara, leah, jess, leo, dan, mia, zac, zoe
- `--temperature <TEMP>`: Sampling temperature (default: 0.6)
- `--top-p <P>`: Top-p sampling parameter
- `--top-k <K>`: Top-k sampling parameter
- `--seed <SEED>`: Random seed for reproducible generation (default: 299792458)
- `--out-file <FILE>`: Output WAV file path (default: out.wav)
- `--which-orpheus-model <MODEL>`: Model variant to use (default: 3b-0.1-ft)
- `--orpheus-model-id <ID>`: Custom model ID from HuggingFace Hub
- `--orpheus-model-file <FILE>`: Local model file path
- `--tokenizer-file <FILE>`: Local tokenizer file path
- `--config-file <FILE>`: Local config file path
- `--revision <REV>`: Model revision/branch (default: main)
- `--use-flash-attn`: Enable Flash Attention for improved performance
- `--verbose-prompt`: Print detailed prompt token information
- `--tracing`: Enable performance tracing

### Example with custom settings

```shell
cargo run --package orpheus-tts --features cuda -- \
  --hf-token YOUR_HF_TOKEN \
  --prompt "Hello world, this is a test of the Orpheus TTS system." \
  --voice leo \
  --temperature 0.8 \
  --out-file speech.wav
```

## Features

- **Multiple Voice Options**: Eight different voices (tara, leah, jess, leo, dan, mia, zac, zoe) with distinct characteristics
- **High-Quality Synthesis**: Based on the Orpheus TTS model for natural-sounding speech
- **Configurable Generation**: Adjustable temperature, top-p, and top-k sampling parameters
- **SNAC Audio Codec**: Advanced neural audio codec for high-fidelity audio generation
- **Flexible Model Loading**: Support for custom models and local files
- **Cross-platform**: Works on CUDA-enabled systems and macOS with Metal acceleration
- **WAV Output**: Direct output to standard WAV format at 24kHz sample rate

## How It Works

The application uses a two-stage process for text-to-speech synthesis:

1. **Text Processing**: The input text is tokenized and processed through the Orpheus language model to generate audio tokens
2. **Audio Synthesis**: The generated audio tokens are decoded using the SNAC (Super Neural Audio Codec) model to produce high-quality PCM audio

The system supports streaming generation with configurable sampling parameters for controlling the quality and characteristics of the generated speech.

## Voice Characteristics

- **tara**: Balanced, clear female voice
- **leah**: Warm, expressive female voice
- **jess**: Energetic, youthful female voice
- **leo**: Strong, confident male voice
- **dan**: Friendly, approachable male voice
- **mia**: Soft, gentle female voice
- **zac**: Deep, authoritative male voice
- **zoe**: Bright, articulate female voice

## Requirements

- HuggingFace API token for model access
- CUDA-compatible GPU (recommended) or macOS with Metal support
- Sufficient disk space for model downloads (models are several GB in size)

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) for the base TTS model
- [candle](https://github.com/huggingface/candle) for the Rust ML framework
- [SNAC](https://github.com/hubertsiuzdak/snac) for the neural audio codec
