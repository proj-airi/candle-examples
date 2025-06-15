# whisper-realtime

![](./docs/demo.svg)

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/whisper-realtime
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package whisper-realtime --features cuda
```

### macOS

```
cargo build --package whisper-realtime --features metal
```

## Run

### NVIDIA CUDA

```shell
cargo run --package whisper-realtime --features cuda
```

### macOS

```shell
cargo run --package whisper-realtime --features metal
```

## Acknowledgements

- [candle/candle-examples/examples/whisper-microphone](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper-microphone)
- [candle/candle-examples/examples/whisper](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)
- [candle/candle-wasm-examples/whisper](https://github.com/huggingface/candle/tree/main/candle-wasm-examples/whisper)
