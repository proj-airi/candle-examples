use std::fmt::Display;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as whisper_model, Config, audio};
use clap::ValueEnum;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use crate::huggingface;

pub enum WhisperModel {
  Normal(whisper_model::model::Whisper),
}

impl WhisperModel {
  pub fn encoder_forward(
    &mut self,
    x: &Tensor,
    flush: bool,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(model) => model.encoder.forward(x, flush),
    }
  }

  pub fn decoder_forward(
    &mut self,
    x: &Tensor,
    encoder_out: &Tensor,
    flush: bool,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(model) => model.decoder.forward(x, encoder_out, flush),
    }
  }

  pub fn decoder_final_linear(
    &self,
    x: &Tensor,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(model) => model.decoder.final_linear(x),
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WhichWhisperModel {
  Tiny,
  #[value(name = "tiny.en")]
  TinyEn,
  Base,
  #[value(name = "base.en")]
  BaseEn,
  Small,
  #[value(name = "small.en")]
  SmallEn,
  Medium,
  #[value(name = "medium.en")]
  MediumEn,
  Large,
  LargeV2,
  LargeV3,
  LargeV3Turbo,
  #[value(name = "distil-medium.en")]
  DistilMediumEn,
  #[value(name = "distil-large-v2")]
  DistilLargeV2,
  #[value(name = "lite-whisper-large-v3-turbo")]
  LiteWhisperLargeV3Turbo,
  #[value(name = "lite-whisper-large-v3-turbo-acc")]
  LiteWhisperLargeV3TurboAcc,
  #[value(name = "lite-whisper-large-v3-turbo-fast")]
  LiteWhisperLargeV3TurboFast,
}

impl Display for WhichWhisperModel {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> std::fmt::Result {
    write!(
      f,
      "{}",
      match self {
        Self::Tiny => "tiny",
        Self::TinyEn => "tiny.en",
        Self::Base => "base",
        Self::BaseEn => "base.en",
        Self::Small => "small",
        Self::SmallEn => "small.en",
        Self::Medium => "medium",
        Self::MediumEn => "medium.en",
        Self::Large => "large",
        Self::LargeV2 => "large-v2",
        Self::LargeV3 => "large-v3",
        Self::LargeV3Turbo => "large-v3-turbo",
        Self::DistilMediumEn => "distil-medium.en",
        Self::DistilLargeV2 => "distil-large-v2",
        Self::LiteWhisperLargeV3Turbo => "lite-whisper-large-v3-turbo",
        Self::LiteWhisperLargeV3TurboAcc => "lite-whisper-large-v3-turbo-acc",
        Self::LiteWhisperLargeV3TurboFast => "lite-whisper-large-v3-turbo-fast",
      }
    )
  }
}

impl WhichWhisperModel {
  pub fn from_str(s: &str) -> Option<Self> {
    match s.to_lowercase().as_str() {
      "tiny" => Some(Self::Tiny),
      "tiny.en" => Some(Self::TinyEn),
      "base" => Some(Self::Base),
      "base.en" => Some(Self::BaseEn),
      "small" => Some(Self::Small),
      "small.en" => Some(Self::SmallEn),
      "medium" => Some(Self::Medium),
      "medium.en" => Some(Self::MediumEn),
      "large" => Some(Self::Large),
      "large-v2" => Some(Self::LargeV2),
      "large-v3" => Some(Self::LargeV3),
      "large-v3-turbo" => Some(Self::LargeV3Turbo),
      "distil-medium.en" => Some(Self::DistilMediumEn),
      "distil-large-v2" => Some(Self::DistilLargeV2),
      "lite-whisper-large-v3-turbo" => Some(Self::LiteWhisperLargeV3Turbo),
      "lite-whisper-large-v3-turbo-acc" => Some(Self::LiteWhisperLargeV3TurboAcc),
      "lite-whisper-large-v3-turbo-fast" => Some(Self::LiteWhisperLargeV3TurboFast),
      _ => None,
    }
  }

  pub const fn model_and_revision(self) -> (&'static str, &'static str) {
    match self {
      Self::Tiny => ("openai/whisper-tiny", "main"),
      Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
      Self::Base => ("openai/whisper-base", "refs/pr/22"),
      Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
      Self::Small => ("openai/whisper-small", "main"),
      Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
      Self::Medium => ("openai/whisper-medium", "main"),
      Self::MediumEn => ("openai/whisper-medium.en", "main"),
      Self::Large => ("openai/whisper-large", "refs/pr/36"),
      Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
      Self::LargeV3 => ("openai/whisper-large-v3", "main"),
      Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
      Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
      Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
      Self::LiteWhisperLargeV3Turbo => ("efficient-speech/lite-whisper-large-v3-turbo", "main"),
      Self::LiteWhisperLargeV3TurboAcc => ("efficient-speech/lite-whisper-large-v3-turbo-acc", "main"),
      Self::LiteWhisperLargeV3TurboFast => ("efficient-speech/lite-whisper-large-v3-turbo-fast", "main"),
    }
  }
}

pub struct WhisperProcessor {
  pub model:       WhisperModel,
  pub tokenizer:   Tokenizer,
  pub config:      Config,
  pub mel_filters: Vec<f32>,
  pub device:      Device,
}

impl WhisperProcessor {
  pub fn new(
    model: WhichWhisperModel,
    device: Device,
  ) -> Result<Self> {
    // Load the Whisper model based on the provided model type
    let api = huggingface::create_hf_api()?;
    let (model_id, revision) = model.model_and_revision();
    let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string()));
    let (openai_large_v3_model_id, openai_large_v3_repo) = WhichWhisperModel::LargeV3.model_and_revision();
    let openai_large_v3_repo = api.repo(Repo::with_revision(openai_large_v3_model_id.to_string(), RepoType::Model, openai_large_v3_repo.to_string()));

    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = match model {
      WhichWhisperModel::LiteWhisperLargeV3Turbo => openai_large_v3_repo.get("tokenizer.json")?,
      WhichWhisperModel::LiteWhisperLargeV3TurboAcc => openai_large_v3_repo.get("tokenizer.json")?,
      WhichWhisperModel::LiteWhisperLargeV3TurboFast => openai_large_v3_repo.get("tokenizer.json")?,
      _ => repo.get("tokenizer.json")?,
    };
    let model_filename = repo.get("model.safetensors")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let name = model_filename.display();
    println!("Loading Whisper model from: {name:?}");

    let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], whisper_model::DTYPE, &device)? };

    let model = WhisperModel::Normal(whisper_model::model::Whisper::load(&var_builder, config.clone())?);

    let mel_bytes = match config.num_mel_bins {
      80 => include_bytes!("../melfilters.bytes").as_slice(),
      128 => include_bytes!("../melfilters128.bytes").as_slice(),
      num_mel_bins => anyhow::bail!("Unsupported number of mel bins: {}", num_mel_bins),
    };

    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <LittleEndian as ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    Ok(Self { model, tokenizer, config, mel_filters, device })
  }

  pub fn transcribe(
    &mut self,
    audio: &[f32],
  ) -> Result<String> {
    // Convert PCM to mel spectrogram
    let mel = audio::pcm_to_mel(&self.config, audio, &self.mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, mel_len / self.config.num_mel_bins), &self.device)?;

    // Run inference
    let audio_features = self.model.encoder_forward(&mel, true)?;
    // Simple greedy decoding
    let tokens = self.decode_greedy(&audio_features)?;

    let text = self
      .tokenizer
      .decode(&tokens, true)
      .map_err(anyhow::Error::msg)?;

    Ok(text)
  }

  fn decode_greedy(
    &mut self,
    audio_features: &Tensor,
  ) -> Result<Vec<u32>> {
    let mut tokens = vec![self.token_id(whisper_model::SOT_TOKEN)?, self.token_id(whisper_model::TRANSCRIBE_TOKEN)?, self.token_id(whisper_model::NO_TIMESTAMPS_TOKEN)?];

    let max_len = 50; // Short sequence for real-time processing

    for i in 0..max_len {
      let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
      let ys = self
        .model
        .decoder_forward(&tokens_t, audio_features, i == 0)?;

      let (_, seq_len, _) = ys.dims3()?;
      let logits = self
        .model
        .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
        .i(0)?
        .i(0)?;

      // Get most likely token
      let logits_v: Vec<f32> = logits.to_vec1()?;
      let next_token = logits_v
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| u32::try_from(i).unwrap())
        .unwrap();

      if next_token == self.token_id(whisper_model::EOT_TOKEN)? {
        break;
      }

      tokens.push(next_token);
    }

    Ok(tokens)
  }

  fn token_id(
    &self,
    token: &str,
  ) -> Result<u32> {
    self
      .tokenizer
      .token_to_id(token)
      .ok_or_else(|| anyhow::anyhow!("Token not found: {}", token))
  }
}
