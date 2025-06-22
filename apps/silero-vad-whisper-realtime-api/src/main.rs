use std::sync::Arc;

use anyhow::Result;
use axum::{
  Json, Router,
  response::IntoResponse,
  routing::{get, post},
};
use candle_core::Device;
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

use crate::{
  router::transcribe_audio,
  vad::VADProcessor,
  whisper::{WhichWhisperModel, WhisperProcessor},
};

mod api;
mod router;
mod audio_manager;
mod vad;
mod whisper;

// Application state
struct AppState {
  vad: Arc<Mutex<VADProcessor>>,
  whisper: Arc<Mutex<WhisperProcessor>>,
  device: Device,
}

impl AppState {
  async fn new() -> Result<Self> {
    // Determine device to use - allow override via environment variable
    let device = if std::env::var("CANDLE_FORCE_CPU").is_ok() {
      candle_core::Device::Cpu
    } else if candle_core::utils::cuda_is_available() {
      candle_core::Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
      candle_core::Device::new_metal(0)?
    } else {
      candle_core::Device::Cpu
    };

    println!("🚀 Using device: {device:?}");

    // Get VAD threshold from environment or use default
    let vad_threshold = std::env::var("VAD_THRESHOLD")
      .ok()
      .and_then(|s| s.parse().ok())
      .unwrap_or(0.3);

    // Get Whisper model from environment or use default
    let whisper_model = match std::env::var("WHISPER_MODEL").as_deref() {
      Ok("tiny") => WhichWhisperModel::Tiny,
      Ok("base") => WhichWhisperModel::Base,
      Ok("small") => WhichWhisperModel::Small,
      Ok("medium") => WhichWhisperModel::Medium,
      Ok("large") => WhichWhisperModel::Large,
      Ok("large-v2") => WhichWhisperModel::LargeV2,
      Ok("large-v3") => WhichWhisperModel::LargeV3,
      _ => WhichWhisperModel::Tiny,
    };

    println!("🎯 VAD threshold: {vad_threshold}");
    println!("🧠 Whisper model: {whisper_model:?}");

    // Initialize VAD and Whisper processors
    let vad = VADProcessor::new(candle_core::Device::Cpu, vad_threshold)?;
    let whisper = WhisperProcessor::new(whisper_model, device.clone())?;

    Ok(Self { vad: Arc::new(Mutex::new(vad)), whisper: Arc::new(Mutex::new(whisper)), device })
  }
}

#[tokio::main]
async fn main() -> Result<()> {
  // Initialize tracing
  tracing_subscriber::fmt::init();

  // Initialize application state
  let state = AppState::new().await?;

  // Build application routes
  let app = Router::new()
    .route("/", get(health_check))
    .route("/v1/audio/transcriptions", post(transcribe_audio))
    .layer(
      ServiceBuilder::new()
        .layer(CorsLayer::permissive())
        .into_inner(),
    )
    .with_state(Arc::new(state));

  // Start server
  let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
  println!("🚀 ASR API server running on http://0.0.0.0:3000");
  println!("📝 Available endpoints:");
  println!("  GET  /                           - Health check");
  println!("  POST /v1/audio/transcriptions    - Audio transcription (OpenAI compatible)");

  axum::serve(listener, app).await?;
  Ok(())
}

// Health check endpoint
async fn health_check() -> impl IntoResponse {
  Json(serde_json::json!({
    "status": "ok",
    "service": "ASR API",
    "version": "1.0.0"
  }))
}
