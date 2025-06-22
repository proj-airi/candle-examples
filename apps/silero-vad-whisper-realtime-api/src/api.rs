use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
  pub text: String,
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
  pub text: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub timestamp: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
  pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
  pub message: String,
  #[serde(rename = "type")]
  pub error_type: String,
  pub param: Option<String>,
  pub code: Option<String>,
}

pub fn default_model() -> String {
  "whisper-1".to_string()
}

pub fn default_response_format() -> String {
  "json".to_string()
}

pub fn default_temperature() -> f32 {
  0.0
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
  /// The audio file object to transcribe
  /// In multipart form, this would be the file field

  /// ID of the model to use. Only whisper-1 is currently available.
  #[serde(default = "default_model")]
  pub model: String,

  /// The language of the input audio
  pub language: Option<String>,

  /// An optional text to guide the model's style or continue a previous audio segment
  pub prompt: Option<String>,

  /// The format of the transcript output
  #[serde(default = "default_response_format")]
  pub response_format: String,

  /// The sampling temperature, between 0 and 1
  #[serde(default = "default_temperature")]
  pub temperature: f32,

  /// Enable streaming response
  #[serde(default)]
  pub stream: bool,
}
