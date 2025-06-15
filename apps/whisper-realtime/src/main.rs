use anyhow::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use candle_transformers::models::whisper::{self as whisper_model, Config, audio};
use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hf_hub::{Repo, RepoType, api::sync::Api};
use rand::{SeedableRng, distr::Distribution};
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

mod multilingual;

const WINDOW_SIZE: usize = 5; // 5 seconds
const OVERLAP_SIZE: usize = 1; // 1 second overlap

pub enum WhisperModel {
  Normal(whisper_model::model::Whisper),
  Quantized(whisper_model::quantized_model::Whisper),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
  Transcribe,
  Translate,
}

pub fn token_id(
  tokenizer: &Tokenizer,
  token: &str,
) -> candle_core::Result<u32> {
  match tokenizer.token_to_id(token) {
    None => candle_core::bail!("no token-id for {token}"),
    Some(id) => Ok(id),
  }
}

impl WhisperModel {
  const fn config(&self) -> &Config {
    match self {
      Self::Normal(m) => &m.config,
      Self::Quantized(m) => &m.config,
    }
  }

  pub fn encoder_forward(
    &mut self,
    x: &Tensor,
    flush: bool,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.encoder.forward(x, flush),
      Self::Quantized(m) => m.encoder.forward(x, flush),
    }
  }

  pub fn decoder_forward(
    &mut self,
    x: &Tensor,
    xa: &Tensor,
    flush: bool,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.decoder.forward(x, xa, flush),
      Self::Quantized(m) => m.decoder.forward(x, xa, flush),
    }
  }

  pub fn decoder_final_linear(
    &self,
    x: &Tensor,
  ) -> candle_core::Result<Tensor> {
    match self {
      Self::Normal(m) => m.decoder.final_linear(x),
      Self::Quantized(m) => m.decoder.final_linear(x),
    }
  }
}

use candle_core::utils::{cuda_is_available, metal_is_available};

pub fn device(cpu: bool) -> Result<Device> {
  if cpu {
    Ok(Device::Cpu)
  } else if cuda_is_available() {
    Ok(Device::new_cuda(0)?)
  } else if metal_is_available() {
    Ok(Device::new_metal(0)?)
  } else {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
      println!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
      println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
    }
    Ok(Device::Cpu)
  }
}

#[derive(Debug, Clone)]
struct DecodingResult {
  tokens:            Vec<u32>,
  text:              String,
  avg_logprob:       f64,
  no_speech_prob:    f64,
  #[allow(dead_code)]
  temperature:       f64,
  compression_ratio: f64,
}

#[derive(Debug, Clone)]
struct Segment {
  start:    f64,
  duration: f64,
  dr:       DecodingResult,
}

struct Decoder {
  model:               WhisperModel,
  rng:                 rand::rngs::StdRng,
  task:                Option<Task>,
  timestamps:          bool,
  verbose:             bool,
  tokenizer:           Tokenizer,
  suppress_tokens:     Tensor,
  sot_token:           u32,
  transcribe_token:    u32,
  translate_token:     u32,
  eot_token:           u32,
  no_speech_token:     u32,
  no_timestamps_token: u32,
  language_token:      Option<u32>,
}

impl Decoder {
  #[allow(clippy::too_many_arguments)]
  fn new(
    model: WhisperModel,
    tokenizer: Tokenizer,
    seed: u64,
    device: &Device,
    language_token: Option<u32>,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
  ) -> Result<Self> {
    let no_timestamps_token = token_id(&tokenizer, whisper_model::NO_TIMESTAMPS_TOKEN)?;

    // Suppress the no_timestamps_token when in timestamps mode.
    // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
    #[allow(clippy::cast_possible_truncation)]
    let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
      .map(|i| {
        if model.config().suppress_tokens.contains(&i) || timestamps && i == no_timestamps_token {
          f32::NEG_INFINITY
        } else {
          0f32
        }
      })
      .collect();

    let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
    let sot_token = token_id(&tokenizer, whisper_model::SOT_TOKEN)?;
    let transcribe_token = token_id(&tokenizer, whisper_model::TRANSCRIBE_TOKEN)?;
    let translate_token = token_id(&tokenizer, whisper_model::TRANSLATE_TOKEN)?;
    let eot_token = token_id(&tokenizer, whisper_model::EOT_TOKEN)?;
    let no_speech_token = whisper_model::NO_SPEECH_TOKENS
      .iter()
      .find_map(|t| token_id(&tokenizer, t).ok());
    let Some(no_speech_token) = no_speech_token else { anyhow::bail!("unable to find any non-speech token") };
    let rng = rand::rngs::StdRng::seed_from_u64(seed);

    Ok(Self {
      model,
      rng,
      task,
      timestamps,
      verbose,
      tokenizer,
      suppress_tokens,
      sot_token,
      transcribe_token,
      translate_token,
      eot_token,
      no_speech_token,
      no_timestamps_token,
      language_token,
    })
  }

  fn decode(
    &mut self,
    mel: &Tensor,
    t: f64,
  ) -> Result<DecodingResult> {
    let model = &mut self.model;
    let audio_features = model.encoder_forward(mel, true)?;
    if self.verbose {
      println!("audio features shape: {:?}", audio_features.shape());
    }

    let sample_len = model.config().max_target_positions / 2;
    let mut sum_logprob = 0f64;
    let mut no_speech_prob = f64::NAN;
    let mut tokens = vec![self.sot_token];
    if let Some(language_token) = self.language_token {
      tokens.push(language_token);
    }

    match self.task {
      Some(Task::Translate) => tokens.push(self.translate_token),
      Some(Task::Transcribe) | None => tokens.push(self.transcribe_token),
    }

    if !self.timestamps {
      tokens.push(self.no_timestamps_token);
    }

    #[allow(clippy::cast_lossless)]
    for i in 0..sample_len {
      let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

      // The model expects a batch dim but this inference loop does not handle
      // it so we add it at this point.
      let tokens_t = tokens_t.unsqueeze(0)?;
      let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

      // Extract the no speech probability on the first iteration by looking at the first
      // token logits and the probability for the according token.
      if i == 0 {
        let logits = model
          .decoder_final_linear(&ys.i(..1)?)?
          .i(0)?
          .i(0)?;
        no_speech_prob = softmax(&logits, 0)?
          .i(self.no_speech_token as usize)?
          .to_scalar::<f32>()? as f64;
      }

      let (_, seq_len, _) = ys.dims3()?;
      #[allow(clippy::cast_precision_loss)]
      let logits = model
        .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
        .i(0)?
        .i(0)?;
      // TODO: Besides suppress tokens, we should apply the heuristics from
      // ApplyTimestampRules, i.e.:
      // - Timestamps come in pairs, except before EOT.
      // - Timestamps should be non-decreasing.
      // - If the sum of the probabilities of timestamps is higher than any other tokens,
      //   only consider timestamps when sampling.
      // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
      let logits = logits.broadcast_add(&self.suppress_tokens)?;

      #[allow(clippy::cast_possible_truncation)]
      let next_token = if t > 0f64 {
        let prs = softmax(&(&logits / t)?, 0)?;
        let logits_v: Vec<f32> = prs.to_vec1()?;
        let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
        distr.sample(&mut self.rng) as u32
      } else {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        #[allow(clippy::cast_possible_truncation)]
        logits_v
          .iter()
          .enumerate()
          .max_by(|(_, u), (_, v)| u.total_cmp(v))
          .map(|(i, _)| i as u32)
          .unwrap()
      };

      tokens.push(next_token);

      #[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
      let prob = softmax(&logits, candle_core::D::Minus1)?
        .i(next_token as usize)?
        .to_scalar::<f32>()? as f64;
      if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
        break;
      }

      sum_logprob += prob.ln();
    }

    let text = self
      .tokenizer
      .decode(&tokens, true)
      .map_err(Error::msg)?;

    #[allow(clippy::cast_precision_loss)]
    let avg_logprob = sum_logprob / tokens.len() as f64;

    Ok(DecodingResult { tokens, text, avg_logprob, no_speech_prob, temperature: t, compression_ratio: f64::NAN })
  }

  fn decode_with_fallback(
    &mut self,
    segment: &Tensor,
  ) -> Result<DecodingResult> {
    for (i, &t) in whisper_model::TEMPERATURES.iter().enumerate() {
      let dr: Result<DecodingResult> = self.decode(segment, t);
      if i == whisper_model::TEMPERATURES.len() - 1 {
        return dr;
      }

      match dr {
        Ok(dr) => {
          let needs_fallback = dr.compression_ratio > whisper_model::COMPRESSION_RATIO_THRESHOLD || dr.avg_logprob < whisper_model::LOGPROB_THRESHOLD;
          if !needs_fallback || dr.no_speech_prob > whisper_model::NO_SPEECH_THRESHOLD {
            return Ok(dr);
          }
        },
        Err(err) => {
          println!("Error running at {t}: {err}");
        },
      }
    }

    unreachable!();
  }

  fn run(
    &mut self,
    mel: &Tensor,
    times: Option<(f64, f64)>,
  ) -> Result<Vec<Segment>> {
    let (_, _, content_frames) = mel.dims3()?;
    let mut seek = 0;
    let mut segments = vec![];

    while seek < content_frames {
      let start = std::time::Instant::now();

      #[allow(clippy::cast_precision_loss)]
      let time_offset = (seek * whisper_model::HOP_LENGTH) as f64 / whisper_model::SAMPLE_RATE as f64;
      let segment_size = usize::min(content_frames - seek, whisper_model::N_FRAMES);
      let mel_segment = mel.narrow(2, seek, segment_size)?;

      #[allow(clippy::cast_precision_loss)]
      let segment_duration = (segment_size * whisper_model::HOP_LENGTH) as f64 / whisper_model::SAMPLE_RATE as f64;
      let dr = self.decode_with_fallback(&mel_segment)?;

      seek += segment_size;

      if dr.no_speech_prob > whisper_model::NO_SPEECH_THRESHOLD && dr.avg_logprob < whisper_model::LOGPROB_THRESHOLD {
        println!("no speech detected, skipping {seek} {dr:?}");
        continue;
      }
      let segment = Segment { start: time_offset, duration: segment_duration, dr };

      if self.timestamps {
        println!("{:.1}s -- {:.1}s", segment.start, segment.start + segment.duration,);

        let mut tokens_to_decode = vec![];
        let mut prev_timestamp_s = 0f32;
        for &token in &segment.dr.tokens {
          if token == self.sot_token || token == self.eot_token {
            continue;
          }

          // The no_timestamp_token is the last before the timestamp ones.
          if token > self.no_timestamps_token {
            #[allow(clippy::cast_precision_loss)]
            let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;

            if !tokens_to_decode.is_empty() {
              let text = self
                .tokenizer
                .decode(&tokens_to_decode, true)
                .map_err(Error::msg)?;
              println!("  {prev_timestamp_s:.1}s-{timestamp_s:.1}s: {text}");

              tokens_to_decode.clear();
            }

            prev_timestamp_s = timestamp_s;
          } else {
            tokens_to_decode.push(token);
          }
        }
        if !tokens_to_decode.is_empty() {
          let text = self
            .tokenizer
            .decode(&tokens_to_decode, true)
            .map_err(Error::msg)?;
          if !text.is_empty() {
            println!("  {prev_timestamp_s:.1}s-...: {text}");
          }

          tokens_to_decode.clear();
        }
      } else {
        match times {
          Some((start, end)) => {
            println!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text);
          },
          None => {
            println!("{:.1}s -- {:.1}s: {}", segment.start, segment.start + segment.duration, segment.dr.text,);
          },
        }
      }
      if self.verbose {
        println!("{seek}: {segment:?}, in {:?}", start.elapsed());
      }

      segments.push(segment);
    }

    Ok(segments)
  }

  const fn set_language_token(
    &mut self,
    language_token: Option<u32>,
  ) {
    self.language_token = language_token;
  }

  fn reset_kv_cache(&mut self) {
    match &mut self.model {
      WhisperModel::Normal(m) => m.reset_kv_cache(),
      WhisperModel::Quantized(m) => m.reset_kv_cache(),
    }
  }

  const fn model(&mut self) -> &mut WhisperModel {
    &mut self.model
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
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
}

impl WhichModel {
  const fn is_multilingual(self) -> bool {
    match self {
      Self::Tiny | Self::Base | Self::Small | Self::Medium | Self::Large | Self::LargeV2 | Self::LargeV3 | Self::LargeV3Turbo | Self::DistilLargeV2 => true,
      Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => false,
    }
  }

  const fn model_and_revision(self) -> (&'static str, &'static str) {
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
    }
  }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Run on CPU rather than on GPU.
  #[arg(long)]
  cpu: bool,

  #[arg(long)]
  model_id: Option<String>,

  /// The model to use, check out available models:
  /// <https://huggingface.co/models?search=whisper>
  #[arg(long)]
  revision: Option<String>,

  /// The model to be used, can be tiny, small, medium.
  #[arg(long, default_value = "tiny.en")]
  model: WhichModel,

  /// The seed to use when generating random samples.
  #[arg(long, default_value_t = 299792458)]
  seed: u64,

  /// Enable tracing (generates a trace-timestamp.json file).
  #[arg(long)]
  tracing: bool,

  #[arg(long)]
  quantized: bool,

  /// Language.
  #[arg(long)]
  language: Option<String>,

  /// Task, when no task is specified, the input tokens contain only the sot token which can
  /// improve things when in no-timestamp mode.
  #[arg(long)]
  task: Option<Task>,

  /// Timestamps mode, this is not fully implemented yet.
  #[arg(long)]
  timestamps: bool,

  /// Print the full `DecodingResult` structure rather than just the text.
  #[arg(long)]
  verbose: bool,

  /// The input device to use.
  #[arg(long)]
  device: Option<String>,
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
  let args = Args::parse();
  let _guard = if args.tracing {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry()
      .with(chrome_layer)
      .init();
    Some(guard)
  } else {
    None
  };

  let device = device(args.cpu)?;
  let (default_model, default_revision) = if args.quantized {
    ("lmz/candle-whisper", "main")
  } else {
    args.model.model_and_revision()
  };

  let default_model = default_model.to_string();
  let default_revision = default_revision.to_string();
  let (model_id, revision) = match (args.model_id, args.revision) {
    (Some(model_id), Some(revision)) => (model_id, revision),
    (Some(model_id), None) => (model_id, "main".to_string()),
    (None, Some(revision)) => (default_model, revision),
    (None, None) => (default_model, default_revision),
  };

  let (config_filename, tokenizer_filename, weights_filename) = {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let (config, tokenizer, model) = if args.quantized {
      let ext = match args.model {
        WhichModel::TinyEn => "tiny-en",
        WhichModel::Tiny => "tiny",
        _ => unimplemented!("no quantized support for {:?}", args.model),
      };

      (repo.get(&format!("config-{ext}.json"))?, repo.get(&format!("tokenizer-{ext}.json"))?, repo.get(&format!("model-{ext}-q80.gguf"))?)
    } else {
      let config = repo.get("config.json")?;
      let tokenizer = repo.get("tokenizer.json")?;
      let model = repo.get("model.safetensors")?;
      (config, tokenizer, model)
    };
    (config, tokenizer, model)
  };

  let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
  let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

  let model = if args.quantized {
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename, &device)?;
    WhisperModel::Quantized(whisper_model::quantized_model::Whisper::load(&vb, config.clone())?)
  } else {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], whisper_model::DTYPE, &device)? };
    WhisperModel::Normal(whisper_model::model::Whisper::load(&vb, config.clone())?)
  };

  let mut decoder = Decoder::new(model, tokenizer.clone(), args.seed, &device, /* language_token */ None, args.task, args.timestamps, args.verbose)?;

  let mel_bytes = match config.num_mel_bins {
    80 => include_bytes!("../melfilters.bytes").as_slice(),
    128 => include_bytes!("../melfilters128.bytes").as_slice(),
    num_mel_bins => anyhow::bail!("unexpected num_mel_bins {num_mel_bins}"),
  };

  let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
  <LittleEndian as ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

  // =====
  // Audio
  // =====

  // Set up the input device and stream with the default input config.
  let host = cpal::default_host();
  let audio_device = match args.device.as_ref() {
    None => host.default_input_device(),
    Some(device) => host
      .input_devices()?
      .find(|x| x.name().is_ok_and(|y| &y == device)),
  }
  .expect("failed to find the audio input device");

  let audio_config = audio_device
    .default_input_config()
    .expect("Failed to get default input config");
  println!("audio config {audio_config:?}");

  let channel_count = audio_config.channels() as usize;
  let in_sample_rate = audio_config.sample_rate().0 as usize;

  #[allow(clippy::cast_precision_loss)]
  let resample_ratio = 16000. / in_sample_rate as f64;

  let mut resampler = rubato::FastFixedIn::new(resample_ratio, 10., rubato::PolynomialDegree::Septic, 1024, 1)?;
  let (tx, rx) = std::sync::mpsc::channel();
  let stream = audio_device.build_input_stream(
    &audio_config.config(),
    move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
      let pcm = pcm
        .iter()
        .step_by(channel_count)
        .copied()
        .collect::<Vec<f32>>();
      if !pcm.is_empty() {
        tx.send(pcm).unwrap();
      }
    },
    move |err| {
      eprintln!("an error occurred on stream: {err}");
    },
    None,
  )?;

  stream.play()?;

  // loop to process the audio data forever (until the user stops the program)
  println!("transcribing audio...");
  let mut buffered_pcm = vec![];
  let mut language_token_set = false;
  while let Ok(pcm) = rx.recv() {
    use rubato::Resampler;

    buffered_pcm.extend_from_slice(&pcm);
    if buffered_pcm.len() < WINDOW_SIZE * in_sample_rate {
      continue;
    }

    let mut resampled_pcm = vec![];
    // resample the audio, one chunk of 1024 samples at a time.
    // in case the audio input failed to produce an exact multiple of 1024 samples,
    // process the remainder on the next iteration of the loop.
    let full_chunks = buffered_pcm.len() / 1024;
    let remainder = buffered_pcm.len() % 1024;

    for chunk in 0..full_chunks {
      let buffered_pcm = &buffered_pcm[chunk * 1024..(chunk + 1) * 1024];
      let pcm = resampler.process(&[&buffered_pcm], None)?;
      resampled_pcm.extend_from_slice(&pcm[0]);
    }

    let pcm = resampled_pcm;
    println!("{} {}", buffered_pcm.len(), pcm.len());

    if remainder == 0 {
      buffered_pcm.clear();
    } else {
      // efficiently copy the remainder to the beginning of the `buffered_pcm` buffer and
      // truncate it.  That's more efficient then allocating a new vector and copying into it
      println!("audio device produced partial chunk with {remainder} samples; processing the remainder on the next iteration of the loop");
      let keep_samples = OVERLAP_SIZE * in_sample_rate;
      let buffered_len = buffered_pcm.len();
      buffered_pcm.copy_within(buffered_len - keep_samples.., 0);
      buffered_pcm.truncate(keep_samples);
    }

    let mel = audio::pcm_to_mel(&config, &pcm, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(mel, (1, config.num_mel_bins, mel_len / config.num_mel_bins), &device)?;

    // on the first iteration, we detect the language and set the language token.
    if !language_token_set {
      let language_token = match (args.model.is_multilingual(), args.language.clone()) {
        (true, None) => Some(multilingual::detect_language(decoder.model(), &tokenizer, &mel)?),
        (false, None) => None,
        (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
          Ok(token_id) => Some(token_id),
          Err(_) => anyhow::bail!("language {language} is not supported"),
        },
        (false, Some(_)) => {
          anyhow::bail!("a language cannot be set for non-multilingual models")
        },
      };
      println!("language_token: {language_token:?}");

      decoder.set_language_token(language_token);
      language_token_set = true;
    }

    decoder.run(&mel, None)?;
    decoder.reset_kv_cache();
  }

  Ok(())
}
