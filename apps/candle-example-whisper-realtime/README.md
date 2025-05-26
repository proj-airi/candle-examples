# candle-example-whisper-realtime

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/candle-example-whisper-realtime
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package candle-example-whisper-realtime --features cuda
```

### macOS

```
cargo build --package candle-example-whisper-realtime --features metal
```
