# candle-example-whisper-mic

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/candle-example-whisper-mic
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package candle-example-whisper-mic --features cuda
```

### macOS

```
cargo build --package candle-example-whisper-mic --features metal
```

## Acknowledgements

- [candle/candle-examples/examples/whisper-microphone](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper-microphone)
- [candle/candle-examples/examples/whisper](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)
- [candle/candle-wasm-examples/whisper](https://github.com/huggingface/candle/tree/main/candle-wasm-examples/whisper)
