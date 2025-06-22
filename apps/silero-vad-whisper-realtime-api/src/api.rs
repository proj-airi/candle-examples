use serde::Serialize;

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
