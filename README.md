<h1 align="center"><a href="https://github.com/huggingface/candle">🤗 candle</a> Examples</h1>

> [!NOTE]
>
> This project is part of (and also associate to) the [Project AIRI](https://github.com/moeru-ai/airi), we aim to build a LLM-driven VTuber like [Neuro-sama](https://www.youtube.com/@Neurosama) (subscribe if you didn't!) if you are interested in, please do give it a try on [live demo](https://airi.moeru.ai).
>
> We use both DuckDB WASM and PGLite for the backbone implementation for memory layer as embedded databases that capable of doing vector search to power up bionic memory systems for AI VTuber and cyber livings. We shared a lot in our [DevLogs](https://airi.moeru.ai/docs/blog/devlog-20250305/) in [DevLog @ 2025.04.06](https://airi.moeru.ai/docs/blog/devlog-20250406/), please read it if you are interested in!
>
> Who are we?
>
> We are a group of currently non-funded talented people made up with computer scientists, experts in multi-modal fields, designers, product managers, and popular open source contributors who loves the goal of where we are heading now.

- [Examples](#examples)
  - [Silero VAD (from files)](#silero-vad-from-files)
  - [Silero VAD (realtime without duration threshold)](#silero-vad-realtime-without-duration-threshold)
  - [Silero VAD (realtime like ChatGPT voice chat)](#silero-vad-realtime-like-chatgpt-voice-chat)
  - [Silero VAD + Whisper (realtime like ChatGPT voice chat)](#silero-vad--whisper-realtime-like-chatgpt-voice-chat)
  - [Whisper Realtime (from microphone)](#whisper-realtime-from-microphone)
- [Development](#development)
  - [NVIDIA CUDA](#nvidia-cuda)
  - [macOS](#macos)
- [Other side projects born from Project AIRI](#other-side-projects-born-from-project-airi)

## Examples

### [Silero VAD (from files)](./apps/silero-vad/README.md)

![](./apps/silero-vad/docs/demo.svg)

### [Silero VAD (realtime without duration threshold)](./apps/silero-vad-realtime-minimum/README.md)

![](./apps/silero-vad-realtime-minimum/docs/demo.svg)

### [Silero VAD (realtime like ChatGPT voice chat)](./apps/silero-vad-realtime/README.md)

![](./apps/silero-vad-realtime/docs/demo.svg)

### [Silero VAD + Whisper (realtime like ChatGPT voice chat)](./apps/silero-vad-whisper-realtime/README.md)

![](./apps/silero-vad-whisper-realtime/docs/demo.svg)

### [Whisper Realtime (from microphone)](./apps/whisper-realtime/README.md)

![](./apps/whisper-realtime/docs/demo.svg)


## Development

```shell
cargo fetch
```

### NVIDIA CUDA

```
cargo build --features cuda
```

### macOS

```
cargo build --features metal
```

## Other side projects born from Project AIRI

- [Awesome AI VTuber](https://github.com/proj-airi/awesome-ai-vtuber): A curated list of AI VTubers and related projects
- [`unspeech`](https://github.com/moeru-ai/unspeech): Universal endpoint proxy server for `/audio/transcriptions` and `/audio/speech`, like LiteLLM but for any ASR and TTS
- [`hfup`](https://github.com/moeru-ai/hfup): tools to help on deploying, bundling to HuggingFace Spaces
- [`xsai-transformers`](https://github.com/moeru-ai/xsai-transformers): Experimental [🤗 Transformers.js](https://github.com/huggingface/transformers.js) provider for [xsAI](https://github.com/moeru-ai/xsai).
- [WebAI: Realtime Voice Chat](https://github.com/proj-airi/webai-realtime-voice-chat): Full example of implementing ChatGPT's realtime voice from scratch with VAD + STT + LLM + TTS.
- [`@proj-airi/drizzle-duckdb-wasm`](https://github.com/moeru-ai/airi/tree/main/packages/drizzle-duckdb-wasm/README.md): Drizzle ORM driver for DuckDB WASM
- [`@proj-airi/duckdb-wasm`](https://github.com/moeru-ai/airi/tree/main/packages/duckdb-wasm/README.md): Easy to use wrapper for `@duckdb/duckdb-wasm`
- [Airi Factorio](https://github.com/moeru-ai/airi-factorio): Allow Airi to play Factorio
- [Factorio RCON API](https://github.com/nekomeowww/factorio-rcon-api): RESTful API wrapper for Factorio headless server console
- [`autorio`](https://github.com/moeru-ai/airi-factorio/tree/main/packages/autorio): Factorio automation library
- [`tstl-plugin-reload-factorio-mod`](https://github.com/moeru-ai/airi-factorio/tree/main/packages/tstl-plugin-reload-factorio-mod): Reload Factorio mod when developing
- [Velin](https://github.com/luoling8192/velin): Use Vue SFC and Markdown to write easy to manage stateful prompts for LLM
- [`demodel`](https://github.com/moeru-ai/demodel): Easily boost the speed of pulling your models and datasets from various of inference runtimes.
- [`inventory`](https://github.com/moeru-ai/inventory): Centralized model catalog and default provider configurations backend service
- [MCP Launcher](https://github.com/moeru-ai/mcp-launcher): Easy to use MCP builder & launcher for all possible MCP servers, just like Ollama for models!
- [🥺 SAD](https://github.com/moeru-ai/sad): Documentation and notes for self-host and browser running LLMs.

## Ackowledgements

- Nice tool [marionebl/svg-term-cli: Share terminal sessions via SVG and CSS](https://github.com/marionebl/svg-term-cli) for converting and rendering [asciinema](https://asciinema.org/) recordings as animated SVG to be able to embed and play in GitHub README.md.
