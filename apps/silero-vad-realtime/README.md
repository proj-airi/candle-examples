# silero-vad-realtime

> This example adds state management and threshold control of both silent and speaking duration on top of how `apps/silero-vad-realtime-minimum` was implemented.
>
> So consider this is a more advanced and feature rich version of the `silero-vad-realtime-minimum` example. Where it suitable in more scenarios instead of detecting only the probability of silence or speech.

![](./docs/demo.svg)

## Getting started

```
git clone https://github.com/proj-airi/candle-examples.git
cd apps/silero-vad-realtime
```

## Build

```
cargo fetch --locked
cargo clean
```

### NVIDIA CUDA

```
cargo build --package silero-vad-realtime
```

### macOS

```
cargo build --package silero-vad-realtime
```

## Run

```shell
cargo run --package silero-vad-realtime
```

## Acknowledgements

- [candle/candle-examples/examples/silero-vad](https://github.com/huggingface/candle/tree/main/candle-examples/examples/silero-vad)
