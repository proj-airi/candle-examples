# ASR API - OpenAI Compatible Audio Transcription Service

🎤 一个兼容OpenAI格式的语音转录API服务，支持实时流式响应(SSE)，集成了Silero VAD和Whisper模型。

## ✨ 功能特性

- 🔄 **兼容OpenAI API**: 完全兼容OpenAI `/v1/audio/transcriptions` 端点格式
- 📡 **Server-Sent Events (SSE)**: 支持流式响应，实时获取转录结果
- 🎯 **语音活动检测**: 集成Silero VAD，智能检测语音片段
- 🧠 **Whisper转录**: 使用Candle框架实现的高效Whisper模型
- 🚀 **高性能**: 支持GPU加速(CUDA/Metal)
- 🌐 **现代Web界面**: 包含完整的测试页面

## 🚀 快速开始

### 1. 启动服务器

```bash
# 进入项目目录
cd apps/asr-api

# 安装依赖并启动
cargo run --release
```

服务器将在 `http://localhost:3000` 启动。

### 2. 测试API

打开浏览器访问测试页面：
```
http://localhost:3000/test.html
```

或者使用curl命令：

```bash
# 基础转录
curl -X POST http://localhost:3000/v1/audio/transcriptions \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1"

# 流式转录
curl -X POST "http://localhost:3000/v1/audio/transcriptions?stream=true" \
  -F "file=@your_audio.wav" \
  -F "model=whisper-1" \
  --no-buffer
```

## 📋 API文档

### POST `/v1/audio/transcriptions`

转录音频文件为文本。

#### 请求参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `file` | File | ✅ | 要转录的音频文件 |
| `model` | String | ❌ | 模型名称 (默认: "whisper-1") |
| `language` | String | ❌ | 音频语言 |
| `prompt` | String | ❌ | 提示文本 |
| `response_format` | String | ❌ | 响应格式 (默认: "json") |
| `temperature` | Float | ❌ | 采样温度 (默认: 0.0) |
| `stream` | Boolean | ❌ | 启用流式响应 (Query参数) |

#### 支持的音频格式

- WAV
- MP3
- FLAC
- M4A
- 以及Symphonia支持的其他格式

#### 响应格式

**标准响应 (JSON)**:
```json
{
  "text": "转录的文本内容"
}
```

**流式响应 (SSE)**:
```
data: {"text": "Processing audio chunk 1 of 4...", "timestamp": 0.5}

data: {"text": "Processing audio chunk 2 of 4...", "timestamp": 1.0}

data: {"text": "转录完成的文本", "timestamp": 2.5}
```

**错误响应**:
```json
{
  "error": {
    "message": "错误描述",
    "type": "invalid_request_error",
    "param": "file",
    "code": null
  }
}
```

## 🛠️ 开发指南

### 项目结构

```
apps/asr-api/
├── src/
│   ├── main.rs           # 主服务器文件
│   ├── vad.rs           # VAD处理器
│   ├── whisper.rs       # Whisper处理器
│   └── audio_manager.rs # 音频缓冲管理
├── melfilters.bytes     # Mel滤波器数据
├── melfilters128.bytes  # 128维Mel滤波器数据
├── test.html           # 测试页面
├── Cargo.toml          # 依赖配置
└── README.md           # 文档
```

### 核心组件

1. **VAD处理器**: 使用Silero VAD模型检测语音活动
2. **Whisper处理器**: 使用Candle实现的Whisper模型进行转录
3. **音频管理器**: 处理音频缓冲和格式转换
4. **Web服务器**: 基于Axum的高性能HTTP服务器

### 自定义配置

可以通过修改 `AppState::new()` 方法来调整以下参数：

- VAD阈值 (默认: 0.3)
- Whisper模型 (默认: Tiny)
- 设备选择 (自动选择GPU/CPU)

### 添加新功能

1. **支持更多音频格式**: 修改 `convert_audio_to_pcm` 函数
2. **自定义VAD参数**: 在 `VADProcessor::new` 中调整参数
3. **更大的Whisper模型**: 在 `WhisperProcessor::new` 中选择不同模型

## 🔧 高级配置

### 环境变量

```bash
# 设置日志级别
export RUST_LOG=debug

# 强制使用CPU
export CANDLE_FORCE_CPU=1
```

### GPU加速

#### CUDA支持
```bash
cargo run --release --features cuda
```

#### Metal支持 (macOS)
```bash
cargo run --release --features metal
```

## 📊 性能优化

### 推荐配置

- **内存**: 最少8GB RAM
- **GPU**: NVIDIA GTX 1060 6GB+ 或 Apple M1+
- **存储**: SSD推荐，用于模型加载

### 批处理优化

对于大量文件处理，建议：

1. 使用更大的Whisper模型获得更好质量
2. 启用GPU加速
3. 调整VAD参数减少误检

## 🚨 常见问题

### Q: 转录准确率不高怎么办？
A: 尝试以下方法：
- 使用更大的Whisper模型 (medium/large)
- 确保音频质量良好 (16kHz采样率)
- 调整VAD阈值
- 提供语言参数

### Q: 服务器启动慢？
A: 首次启动需要下载模型文件，这是正常现象。模型会缓存到本地。

### Q: 支持实时语音输入吗？
A: 目前只支持文件上传，实时语音输入可以参考 `silero-vad-whisper-realtime` 项目。

### Q: 如何批量处理文件？
A: 可以编写脚本调用API，或者扩展当前代码支持批处理端点。

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交改动
4. 发起Pull Request

## 📄 许可证

本项目采用与父项目相同的许可证。

## 🙏 致谢

- [Candle](https://github.com/huggingface/candle) - 高性能ML框架
- [Axum](https://github.com/tokio-rs/axum) - 现代Web框架
- [OpenAI](https://openai.com/) - API设计参考
- [Silero VAD](https://github.com/snakers4/silero-vad) - VAD模型

---

🎯 **提示**: 第一次运行时会自动下载模型文件，请确保网络连接正常。 
