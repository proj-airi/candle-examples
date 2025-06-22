use std::{fs::File, io::prelude::*, time::Instant};

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, utils};
use candle_nn::VarBuilder;
use candle_transformers::{
  generation::LogitsProcessor,
  models::{
    llama::{Cache, Llama as LlamaModel, LlamaConfig},
    snac::{Config as SnacConfig, Model as SnacModel},
  },
};
use clap::Parser;
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

// https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/realtime_streaming_example/main.py#L43
const STOP_TOKEN_ID: u32 = 128258;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Voice {
  #[value(name = "tara")]
  Tara,
  #[value(name = "leah")]
  Leah,
  #[value(name = "jess")]
  Jess,
  #[value(name = "leo")]
  Leo,
  #[value(name = "dan")]
  Dan,
  #[value(name = "mia")]
  Mia,
  #[value(name = "zac")]
  Zac,
  #[value(name = "zoe")]
  Zoe,
}

impl Voice {
  fn as_str(&self) -> &'static str {
    match self {
      Voice::Tara => "tara",
      Voice::Leah => "leah",
      Voice::Jess => "jess",
      Voice::Leo => "leo",
      Voice::Dan => "dan",
      Voice::Mia => "mia",
      Voice::Zac => "zac",
      Voice::Zoe => "zoe",
    }
  }
}

#[derive(Parser)]
struct Args {
  #[arg(long)]
  cpu:                 bool,
  #[arg(long)]
  tracing:             bool,
  #[arg(long)]
  verbose_prompt:      bool,
  #[arg(long, default_value = "Hey, how are you doing today?")]
  prompt:              String,
  #[arg(long, default_value_t = 0.6)]
  temperature:         f64,
  #[arg(long)]
  top_p:               Option<f64>,
  #[arg(long)]
  top_k:               Option<usize>,
  #[arg(long, default_value_t = 299792458)]
  seed:                u64,
  #[arg(long)]
  orpheus_model_id:    Option<String>,
  #[arg(long)]
  revision:            Option<String>,
  #[arg(long)]
  orpheus_model_file:  Option<String>,
  #[arg(long)]
  tokenizer_file:      Option<String>,
  #[arg(long)]
  config_file:         Option<String>,
  #[arg(long, default_value = "out.wav")]
  out_file:            String,
  #[arg(long, default_value = "3b-0.1-ft")]
  which_orpheus_model: WhichOrpheusModel,
  #[arg(long, default_value = "tara")]
  voice:               Voice,
  #[arg(long)]
  use_flash_attn:      bool,
  #[arg(long)]
  hf_token:            String,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum WhichOrpheusModel {
  #[value(name = "3b-0.1-ft")]
  ThreeB0_1Ft,
}

fn hub_load_safetensors(
  repo: &hf_hub::api::sync::ApiRepo,
  json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
  let json_file_path = repo
    .get(json_file)
    .map_err(candle_core::Error::wrap)?;

  let json_file = File::open(&json_file_path)?;
  let json: serde_json::Value = serde_json::from_reader(json_file).map_err(candle_core::Error::wrap)?;
  let weight_map = match json.get("weight_map") {
    None => anyhow::bail!("no weight_map exists inside {:?}", json_file_path),
    Some(serde_json::Value::Object(map)) => map,
    Some(_) => anyhow::bail!("weight_map isn't a object"),
  };

  let mut safetensors_files = std::collections::HashSet::new();
  for value in weight_map.values() {
    if let Some(file) = value.as_str() {
      safetensors_files.insert(file.to_string());
    }
  }

  let safetensors_files = safetensors_files
    .iter()
    .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
    .collect::<Result<Vec<_>, _>>()?;

  Ok(safetensors_files)
}

fn load_snac_model(
  hf_token: &str,
  device: &Device,
) -> Result<SnacModel> {
  let api = hf_hub::api::sync::ApiBuilder::new()
    .with_token(Some(hf_token.to_string()))
    .build()?;

  let snac_model = api.model("hubertsiuzdak/snac_24khz".to_string());
  let config = snac_model.get("config.json")?;
  let config: SnacConfig = serde_json::from_reader(File::open(config)?)?;
  let candle_snac_model = api.model("lmz/candle-snac".to_string());
  let model = candle_snac_model.get("snac_24khz.safetensors")?;
  let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, device)? };

  let model = SnacModel::new(&config, var_builder)?;
  Ok(model)
}

pub trait Sample {
  fn to_i16(&self) -> i16;
}

impl Sample for f32 {
  fn to_i16(&self) -> i16 {
    (self.clamp(-1.0, 1.0) * 32767.0) as i16
  }
}

impl Sample for f64 {
  fn to_i16(&self) -> i16 {
    (self.clamp(-1.0, 1.0) * 32767.0) as i16
  }
}

impl Sample for i16 {
  fn to_i16(&self) -> i16 {
    *self
  }
}

pub fn write_pcm_as_wav<W: Write, S: Sample>(
  w: &mut W,
  samples: &[S],
  sample_rate: u32,
) -> std::io::Result<()> {
  let len = 12u32; // header
  let len = len + 24u32; // fmt
  let len = len + samples.len() as u32 * 2 + 8; // data
  let n_channels = 1u16;
  let bytes_per_second = sample_rate * 2 * n_channels as u32;
  w.write_all(b"RIFF")?;
  w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
  w.write_all(b"WAVE")?;

  // Format block
  w.write_all(b"fmt ")?;
  w.write_all(&16u32.to_le_bytes())?; // block len minus 8 bytes
  w.write_all(&1u16.to_le_bytes())?; // PCM
  w.write_all(&n_channels.to_le_bytes())?; // one channel
  w.write_all(&sample_rate.to_le_bytes())?;
  w.write_all(&bytes_per_second.to_le_bytes())?;
  w.write_all(&2u16.to_le_bytes())?; // 2 bytes of data per sample
  w.write_all(&16u16.to_le_bytes())?; // bits per sample

  // Data block
  w.write_all(b"data")?;
  w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
  for sample in samples.iter() {
    w.write_all(&sample.to_i16().to_le_bytes())?
  }
  Ok(())
}

struct Model {
  model:            LlamaModel,
  tokenizer:        Tokenizer,
  logits_processor: LogitsProcessor,
  cache:            Cache,
  device:           Device,
  verbose_prompt:   bool,
  snac_model:       SnacModel,
  out_file:         String,
  voice:            Voice,
}

impl Model {
  fn load(args: Args) -> Result<Self> {
    let start_time = Instant::now();
    let hf_token = args.hf_token.clone();

    let api = hf_hub::api::sync::ApiBuilder::new()
      .with_token(Some(hf_token.clone()))
      .build()?;

    let model_id = match args.orpheus_model_id {
      Some(model_id) => model_id.to_string(),
      None => match args.which_orpheus_model {
        WhichOrpheusModel::ThreeB0_1Ft => "canopylabs/orpheus-3b-0.1-ft".to_string(),
      },
    };
    let revision = match args.revision {
      Some(r) => r,
      None => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(model_id, hf_hub::RepoType::Model, revision));
    let model_file = match args.orpheus_model_file {
      Some(m) => vec![m.into()],
      None => match args.which_orpheus_model {
        WhichOrpheusModel::ThreeB0_1Ft => hub_load_safetensors(&repo, "model.safetensors.index.json")?,
      },
    };
    let config = match args.config_file {
      Some(c) => c.into(),
      None => repo.get("config.json")?,
    };
    let tokenizer = match args.tokenizer_file {
      Some(t) => t.into(),
      None => repo.get("tokenizer.json")?,
    };

    println!("retrieved the files in {:?}", start_time.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let start_time = Instant::now();
    let device = if args.cpu {
      candle_core::Device::Cpu
    } else if candle_core::utils::cuda_is_available() {
      candle_core::Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
      candle_core::Device::new_metal(0)?
    } else {
      candle_core::Device::Cpu
    };
    let dtype = device.bf16_default_to_f32();

    let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&model_file, dtype, &device)? };
    let config: LlamaConfig = serde_json::from_reader(File::open(config)?)?;
    let config = config.into_config(args.use_flash_attn);
    let model = LlamaModel::load(var_builder, &config)?;

    let logits_processor = {
      use candle_transformers::generation::{LogitsProcessor, Sampling};

      let temperature = args.temperature;
      let sample = if args.temperature < 0. {
        Sampling::ArgMax
      } else {
        match (args.top_k.as_ref(), args.top_p.as_ref()) {
          (None, None) => Sampling::All { temperature },
          (Some(&top_k), None) => Sampling::TopK { k: top_k, temperature },
          (None, Some(&top_p)) => Sampling::TopP { p: top_p, temperature },
          (Some(&top_k), Some(&top_p)) => Sampling::TopKThenTopP { k: top_k, p: top_p, temperature },
        }
      };

      LogitsProcessor::from_sampling(args.seed, sample)
    };

    println!("load the model in {:?}", start_time.elapsed());

    let cache = Cache::new(true, dtype, &config, &device)?;
    let snac_model = load_snac_model(hf_token.clone().as_str(), &device)?;

    Ok(Self {
      model,
      tokenizer,
      logits_processor,
      cache,
      device,
      verbose_prompt: args.verbose_prompt,
      snac_model,
      out_file: args.out_file,
      voice: args.voice,
    })
  }

  fn run(
    &mut self,
    prompt: &str,
  ) -> Result<()> {
    let device = &self.device;
    let prompt = format!("{voice}: {prompt}", voice = self.voice.as_str());
    let tokens = self
      .tokenizer
      .encode(prompt, true)
      .map_err(E::msg)?;

    // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/engine_class.py#L82
    let mut tokens = [&[128259], tokens.get_ids(), &[128009, 128260, 128261, 128257]].concat();
    if self.verbose_prompt {
      println!("prompt tokens: {tokens:?}");
    }
    let mut cache = self.cache.clone();

    println!("starting inference loop");

    let mut index_pos = 0;
    let mut audio_tokens = vec![];

    for index in 0..2000 {
      println!("index {index} at position {index_pos}");

      let (context_size, context_index) = if index > 0 {
        (1, index_pos)
      } else {
        (tokens.len(), 0)
      };

      let context = &tokens[tokens.len().saturating_sub(context_size)..];
      let input = Tensor::new(context, device)?.unsqueeze(0)?;
      let logits = self
        .model
        .forward(&input, context_index, &mut cache)?;
      let logits = logits.squeeze(0)?;
      index_pos += context.len();

      let next_token = self.logits_processor.sample(&logits)?;
      if let Some(tok) = self.tokenizer.id_to_token(next_token) {
        match tok.strip_prefix("<custom_token_") {
          Some(tok) => match tok.strip_suffix('>') {
            Some(tok) => {
              let tok = tok.parse::<u32>()?;
              // https://github.com/canopyai/Orpheus-TTS/blob/df0b0d96685dd21885aef7f900ee7f705c669e94/orpheus_tts_pypi/orpheus_tts/decoder.py#L86C35-L86C63
              let tok = tok - 10 - ((audio_tokens.len() as u32 % 7) * 4096);
              audio_tokens.push(tok);
            },
            None => {
              println!("{index}: unexpected custom token {next_token} {tok}");
            },
          },
          None => {
            println!("{index}: unexpected token {next_token} {tok}");
          },
        }
      }

      if next_token == STOP_TOKEN_ID {
        println!("reached stop token at index {index}");
        break;
      }

      tokens.push(next_token);
    }

    println!("generated {} audio tokens", audio_tokens.len());

    let mut codes0 = vec![];
    let mut codes1 = vec![];
    let mut codes2 = vec![];

    for audio_tokens in audio_tokens.chunks_exact(7) {
      codes0.push(audio_tokens[0]);
      for i in [1, 4] {
        codes1.push(audio_tokens[i]);
      }
      for i in [2, 3, 5, 6] {
        codes2.push(audio_tokens[i]);
      }
    }
    let codes0 = Tensor::new(codes0, device)?.unsqueeze(0)?;
    let codes1 = Tensor::new(codes1, device)?.unsqueeze(0)?;
    let codes2 = Tensor::new(codes2, device)?.unsqueeze(0)?;
    let pcm = self
      .snac_model
      .decode(&[&codes0, &codes1, &codes2])?;

    println!("decoded to pcm {pcm:?}");
    let mut output = std::fs::File::create(&self.out_file)?;
    let pcm = pcm.i(0)?.i(0)?.to_vec1::<f32>()?;
    write_pcm_as_wav(&mut output, &pcm, 24000)?;
    Ok(())
  }
}

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

  println!("avx: {}, neon: {}, simd128: {}, f16c: {}", utils::with_avx(), utils::with_neon(), utils::with_simd128(), utils::with_f16c());

  let prompt = args.prompt.clone();
  let mut model = Model::load(args)?;
  model.run(&prompt)?;
  Ok(())
}
