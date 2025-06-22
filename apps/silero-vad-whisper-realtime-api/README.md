# ASR API - OpenAI Compatible Audio Transcription Service

🎤 An OpenAI-compatible speech transcription API service with real-time streaming response (SSE), integrated with Silero VAD and Whisper models.

## ✨ Features

- 🔄 **OpenAI API Compatible**: Full compatibility with OpenAI `/v1/audio/transcriptions` endpoint format
- 📡 **Server-Sent Events (SSE)**: Supports streaming responses for real-time transcription results
- 🎯 **Voice Activity Detection**: Integrated with Silero VAD for intelligent speech segment detection
- 🧠 **Whisper Transcription**: High-performance Whisper model implementation using Candle framework
- 🚀 **High Performance**: Supports GPU acceleration (CUDA/Metal)
- 🌐 **Modern Web Interface**: Includes complete testing page

## 🚀 Quick Start

### 1. Start the Server

```bash
# Navigate to project directory
cd apps/asr-api

# Install dependencies and start
cargo run --release
```

The server will start at `http://localhost:3000`.

### 2. Test API

```bash
# Basic transcription
curl -X POST http://localhost:3000/v1/audio/transcriptions \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1"

# Streaming transcription
curl -X POST "http://localhost:3000/v1/audio/transcriptions?stream=true" \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1" \
  --no-buffer
```

## 📋 API Documentation

### POST `/v1/audio/transcriptions`

Transcribe audio file to text.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✅ | Audio file to transcribe |
| `model` | String | ❌ | Model name (default: "whisper-1") |
| `language` | String | ❌ | Audio language |
| `prompt` | String | ❌ | Prompt text |
| `response_format` | String | ❌ | Response format (default: "json") |
| `temperature` | Float | ❌ | Sampling temperature (default: 0.0) |
| `stream` | Boolean | ❌ | Enable streaming response (Query parameter) |

#### Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A
- And other formats supported by Symphonia

#### Response Format

**Standard Response (JSON)**:
```json
{
  "text": "Transcribed text content"
}
```

**Streaming Response (SSE)**:
```
data: {"text": "Processing audio chunk 1 of 4...", "timestamp": 0.5}

data: {"text": "Processing audio chunk 2 of 4...", "timestamp": 1.0}

data: {"text": "Completed transcription text", "timestamp": 2.5}
```

**Error Response**:
```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "file",
    "code": null
  }
}
```

## 🛠️ Development Guide

### Project Structure

```
apps/asr-api/
├── src/
│   ├── main.rs           # Main server file
│   ├── vad.rs           # VAD processor
│   ├── whisper.rs       # Whisper processor
│   └── audio_manager.rs # Audio buffer management
├── melfilters.bytes     # Mel filter data
├── melfilters128.bytes  # 128-dim Mel filter data
├── test.html           # Test page
├── Cargo.toml          # Dependencies configuration
└── README.md           # Documentation
```

### Core Components

1. **VAD Processor**: Uses Silero VAD model for voice activity detection
2. **Whisper Processor**: Uses Candle-implemented Whisper model for transcription
3. **Audio Manager**: Handles audio buffering and format conversion
4. **Web Server**: High-performance HTTP server based on Axum

### Custom Configuration

You can adjust the following parameters by modifying the `AppState::new()` method:

- VAD threshold (default: 0.3)
- Whisper model (default: Tiny)
- Device selection (auto-select GPU/CPU)

### Adding New Features

1. **Support more audio formats**: Modify `convert_audio_to_pcm` function
2. **Custom VAD parameters**: Adjust parameters in `VADProcessor::new`
3. **Larger Whisper models**: Select different models in `WhisperProcessor::new`

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Set log level
export RUST_LOG=debug

# Force CPU usage
export CANDLE_FORCE_CPU=1
```

### GPU Acceleration

#### CUDA Support
```bash
cargo run --release --features cuda
```

#### Metal Support (macOS)
```bash
cargo run --release --features metal
```

## 📊 Performance Optimization

### Recommended Configuration

- **Memory**: Minimum 8GB RAM
- **GPU**: NVIDIA GTX 1060 6GB+ or Apple M1+
- **Storage**: SSD recommended for model loading

### Batch Processing Optimization

For processing large numbers of files, consider:

1. Use larger Whisper models for better quality
2. Enable GPU acceleration
3. Adjust VAD parameters to reduce false positives

## 🚨 FAQ

### Q: How to improve transcription accuracy?
A: Try the following methods:
- Use larger Whisper models (medium/large)
- Ensure good audio quality (16kHz sampling rate)
- Adjust VAD threshold
- Provide language parameter

### Q: Server starts slowly?
A: First startup requires downloading model files, which is normal. Models are cached locally.

### Q: Does it support real-time voice input?
A: Currently only supports file upload. For real-time voice input, refer to the `silero-vad-whisper-realtime` project.

### Q: How to batch process files?
A: You can write scripts to call the API, or extend the current code to support batch processing endpoints.

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork the project
2. Create feature branch
3. Commit changes
4. Submit Pull Request

## 📄 License

This project uses the same license as the parent project.

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) - High-performance ML framework
- [Axum](https://github.com/tokio-rs/axum) - Modern web framework
- [OpenAI](https://openai.com/) - API design reference
- [Silero VAD](https://github.com/snakers4/silero-vad) - VAD model

---

🎯 **Tip**: Model files will be automatically downloaded on first run. Please ensure stable network connection. 
