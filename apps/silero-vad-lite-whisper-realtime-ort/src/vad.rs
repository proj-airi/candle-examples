use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use hf_hub::Repo;
use ort::{
  execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider},
  session::{Session, builder::GraphOptimizationLevel},
  util::Mutex,
  value::Tensor,
};
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Serialize, Deserialize, Clone)]
pub struct SileroVADResult {
  pub output: Vec<f32>, // Speech probability output
  pub state:  Vec<f32>, // Updated state for next inference
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SileroVADInput {
  pub input: Vec<f32>, // Audio input buffer
  pub sr:    i64,      // Sample rate
  pub state: Vec<f32>, // Current state
}

pub struct SileroVADModel {
  session: Arc<Mutex<Session>>,
}

impl SileroVADModel {
  pub fn new() -> Result<Self> {
    let model_id = "onnx-community/silero-vad";
    let revision = "main";

    let cache_api = hf_hub::Cache::from_env();
    let cache_repo = cache_api.repo(Repo::with_revision(model_id.into(), hf_hub::RepoType::Model, revision.into()));

    let api = hf_hub::api::sync::ApiBuilder::new().build()?;
    let repo = api.repo(hf_hub::Repo::with_revision(model_id.into(), hf_hub::RepoType::Model, revision.into()));

    let model_path_sub_name = "onnx/model.onnx";
    let model_path = match cache_repo.get(model_path_sub_name) {
      Some(path) => path,
      None => repo.download(model_path_sub_name)?,
    };

    let session = Self::create_optimized_session(model_path.clone())?;

    Ok(Self { session: Arc::new(Mutex::new(session)) })
  }

  /// Create an optimized ONNX session with hardware acceleration
  fn create_optimized_session(model_path: PathBuf) -> Result<Session> {
    let builder = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(1)?;

    let session = builder
      .with_execution_providers(vec![
        CUDAExecutionProvider::default()
          .with_device_id(0)
          .build(),
        CoreMLExecutionProvider::default().build(),
        DirectMLExecutionProvider::default()
          .with_device_id(0)
          .build(),
        CPUExecutionProvider::default().build(),
      ])?
      .commit_from_file(model_path)?;
    info!("VAD model loaded successfully");

    Ok(session)
  }

  /// Stateless inference that matches JavaScript interface
  /// Returns both output and updated state like the JS version
  pub fn inference(
    &self,
    input_data: SileroVADInput,
  ) -> Result<SileroVADResult> {
    // Validate input dimensions
    if input_data.state.len() != 2 * 1 * 128 {
      return Err(anyhow::anyhow!("State must have 256 elements (2*1*128), got {}", input_data.state.len()));
    }

    // Create input tensors for the ONNX model
    let inputs = vec![("input", Tensor::from_array((vec![1, input_data.input.len()], input_data.input.clone()))?.into_dyn()), ("sr", Tensor::from_array(([1], vec![input_data.sr]))?.into_dyn()), ("state", Tensor::from_array((vec![2, 1, 128], input_data.state.clone()))?.into_dyn())];

    // Run inference and extract data while session is still locked
    let (state_data, speech_data) = {
      let mut session = self.session.lock();
      let outputs = session.run(inputs)?;

      // Extract and clone the data immediately while session is locked
      let (_state_shape, state_slice) = outputs[1].try_extract_tensor::<f32>()?;
      let (_speech_shape, speech_slice) = outputs[0].try_extract_tensor::<f32>()?;

      // Clone the data to owned vectors before the session lock is released
      (state_slice.to_vec(), speech_slice.to_vec())
    };

    Ok(SileroVADResult { output: speech_data, state: state_data })
  }
}

/// A pipeline for Silero VAD that manages the model and its state.
///
/// This provides a simple, stateful-feeling interface to the stateless ONNX model,
/// making it easy to use in a streaming context.
pub struct SileroVADPipeline {
  model:       SileroVADModel,
  state:       Vec<f32>,
  sample_rate: i64,
  threshold:   f32,
}

impl SileroVADPipeline {
  /// Creates a new VAD pipeline.
  ///
  /// # Arguments
  /// * `sample_rate` - The sample rate of the audio chunks being processed.
  /// * `threshold` - The probability threshold for detecting speech (e.g., 0.5).
  pub fn new(
    sample_rate: u32,
    threshold: f32,
  ) -> Result<Self> {
    let model = SileroVADModel::new()?;
    // The initial state for the Silero VAD model is a tensor of shape (2, 1, 128) filled with zeros.
    let state = vec![0.0; 2 * 1 * 128];
    Ok(Self { model, state, sample_rate: i64::from(sample_rate), threshold })
  }

  /// Processes a single audio chunk and returns the speech probability.
  ///
  /// The internal state is updated after each call, ready for the next chunk.
  /// The chunk size must be one of the sizes supported by Silero VAD (e.g., 512, 1024 for 16kHz).
  pub fn process_chunk(
    &mut self,
    chunk: &[f32],
  ) -> Result<f32> {
    let input_data = SileroVADInput { input: chunk.to_vec(), sr: self.sample_rate, state: self.state.clone() };

    let result = self.model.inference(input_data)?;

    // Update the state for the next inference call
    self.state = result.state;

    // The model output is a Vec with one element: the speech probability.
    let speech_prob = result.output.first().copied().unwrap_or(0.0);

    Ok(speech_prob)
  }

  /// Determines if speech is present based on the probability and configured threshold.
  pub fn is_speech(
    &self,
    prob: f32,
  ) -> bool {
    prob >= self.threshold
  }
}
