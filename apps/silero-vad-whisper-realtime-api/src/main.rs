use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use axum::{
  Json, Router,
  response::IntoResponse,
  routing::{get, post},
};
use candle_core::Device;
use tokio::sync::{Mutex, RwLock};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

use crate::{
  router::transcribe_audio,
  vad::VADProcessor,
  whisper::{WhichWhisperModel, WhisperProcessor},
};

mod api;
mod audio_manager;
mod router;
mod vad;
mod whisper;

// Application state with dynamic model loading
struct AppState {
  vad: Arc<Mutex<VADProcessor>>,
  device: Device,
  // Use RwLock for read-heavy workload (checking cache)
  whisper_models: Arc<RwLock<HashMap<String, Arc<Mutex<WhisperProcessor>>>>>,
}

impl AppState {
  async fn new() -> Result<Self> {
    // Determine device to use
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

    println!("🎯 VAD threshold: {vad_threshold}");

    // Initialize VAD processor (always use CPU for VAD)
    let vad = VADProcessor::new(candle_core::Device::Cpu, vad_threshold)?;

    Ok(Self {
      vad: Arc::new(Mutex::new(vad)),
      device,
      whisper_models: Arc::new(RwLock::new(HashMap::new())),
    })
  }

  // Get or create Whisper processor for the specified model
  pub async fn get_whisper_processor(
    &self,
    model_name: &str,
  ) -> Result<Arc<Mutex<WhisperProcessor>>> {
    // First, try to read from cache
    {
      let models = self.whisper_models.read().await;
      if let Some(processor) = models.get(model_name) {
        println!("🔄 Using cached Whisper model: {}", model_name);
        return Ok(processor.clone());
      }
    }

    // If not in cache, create new model with timing
    let loading_start = std::time::Instant::now();
    println!("🧠 Loading new Whisper model: {}", model_name);

    let whisper_model = Self::parse_model_name(model_name)?;
    let processor = Arc::new(Mutex::new(WhisperProcessor::new(whisper_model, self.device.clone())?));

    let loading_time = loading_start.elapsed();

    // Add to cache
    {
      let mut models = self.whisper_models.write().await;
      models.insert(model_name.to_string(), processor.clone());
    }

    println!("✅ Whisper model loaded and cached: {} ({:.2}ms)", model_name, loading_time.as_secs_f64() * 1000.0);
    Ok(processor)
  }

  // Parse model name string to WhichWhisperModel enum
  fn parse_model_name(model_name: &str) -> Result<WhichWhisperModel> {
    match model_name.to_lowercase().as_str() {
      "tiny" => Ok(WhichWhisperModel::Tiny),
      "tiny.en" => Ok(WhichWhisperModel::TinyEn),
      "base" => Ok(WhichWhisperModel::Base),
      "base.en" => Ok(WhichWhisperModel::BaseEn),
      "small" => Ok(WhichWhisperModel::Small),
      "small.en" => Ok(WhichWhisperModel::SmallEn),
      "medium" => Ok(WhichWhisperModel::Medium),
      "medium.en" => Ok(WhichWhisperModel::MediumEn),
      "large" => Ok(WhichWhisperModel::Large),
      "large-v2" => Ok(WhichWhisperModel::LargeV2),
      "large-v3" => Ok(WhichWhisperModel::LargeV3),
      "large-v3-turbo" => Ok(WhichWhisperModel::LargeV3Turbo),
      "distil-medium.en" => Ok(WhichWhisperModel::DistilMediumEn),
      "distil-large-v2" => Ok(WhichWhisperModel::DistilLargeV2),
      _ => anyhow::bail!("Unsupported Whisper model: {}. Supported models: tiny, base, small, medium, large, large-v2, large-v3", model_name),
    }
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
