# silero-vad

![](./docs/demo.svg)

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/silero-vad
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package silero-vad
```

### macOS

```
cargo build --package silero-vad
```

## Run

```shell
rec -t raw -r 48000 -b 16 -c 1 -e signed-integer .temp/recording.raw trim 0 5
```

```shell
sox -t raw -r 48000 -b 16 -c 1 -e signed-integer .temp/recording.raw -t raw -r 16000 -b 16 -c 1 -e signed-integer .temp/recording_16k.raw
```

```shell
cargo run --package silero-vad -- --file .temp/recording_16k.raw
```

## Acknowledgements

- [candle/candle-examples/examples/silero-vad](https://github.com/huggingface/candle/tree/main/candle-examples/examples/silero-vad)
