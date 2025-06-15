use std::{collections::HashMap, fs::File, path::PathBuf};

use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};
use candle_onnx::simple_eval;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Run on CPU or GPU
  #[arg(long)]
  cpu:         bool,
  /// Enable tracing (generates a trace-timestamp.json file).
  #[arg(long)]
  tracing:     bool,
  /// Input file path (if not provided, reads from stdin)
  #[arg(long)]
  file:        Option<PathBuf>,
  /// Sample rate for the audio input (VAD model expects 16000 or 8000).
  #[arg(long, default_value = "16000")]
  sample_rate: u32,
}

fn load_model() -> Result<candle_onnx::onnx::ModelProto> {
  let api = hf_hub::api::sync::Api::new()?;
  let model_path = api
    .model("onnx-community/silero-vad".into())
    .get("onnx/model.onnx")?;

  println!("Loading model from: {}", model_path.display());
  let model = candle_onnx::read_file(model_path)?;

  Ok(model)
}

/// Iterator that reads consecutive frames of i16 values and converts to f32
struct AudioFrameReader<R> {
  reader:     R,
  buffer:     Box<[u8]>,
  bytes_read: usize,
  is_eof:     bool,
}

impl<R> AudioFrameReader<R> {
  fn new(
    reader: R,
    frame_size: usize,
  ) -> Self {
    Self {
      reader,
      buffer: vec![0; frame_size * std::mem::size_of::<i16>()].into_boxed_slice(),
      bytes_read: 0,
      is_eof: false,
    }
  }
}

impl<R: std::io::Read> Iterator for AudioFrameReader<R> {
  type Item = std::io::Result<Vec<f32>>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.is_eof {
      return None;
    }

    self.bytes_read += match self
      .reader
      .read(&mut self.buffer[self.bytes_read..])
    {
      std::result::Result::Ok(0) => {
        self.is_eof = true;
        0
      },
      std::result::Result::Ok(n) => n,
      Err(e) => return Some(Err(e)),
    };

    if self.is_eof || self.bytes_read == self.buffer.len() {
      let audio_data = self.buffer[..self.bytes_read]
        .chunks(2)
        .map(|bytes| match bytes {
          [a, b] => i16::from_le_bytes([*a, *b]),
          _ => unreachable!(),
        })
        .map(|sample| f32::from(sample) / f32::from(i16::MAX))
        .collect();

      self.bytes_read = 0;
      Some(std::result::Result::Ok(audio_data))
    } else {
      self.next()
    }
  }
}

struct VADState {
  model:        candle_onnx::onnx::ModelProto,
  frame_size:   usize,
  context_size: usize,
  sample_rate:  Tensor,
  state:        Tensor,
  context:      Tensor,
  device:       Device,
}

impl VADState {
  fn new(
    model: candle_onnx::onnx::ModelProto,
    sample_rate_value: i64,
    device: Device,
  ) -> Result<Self> {
    let (frame_size, context_size) = match sample_rate_value {
      16000 => (512, 64),
      8000 => (256, 32),
      _ => return Err(anyhow::anyhow!("Unsupported sample rate: {}", sample_rate_value)),
    };

    Ok(Self {
      model,
      frame_size,
      context_size,
      sample_rate: Tensor::new(sample_rate_value, &device)?,
      state: Tensor::zeros((2, 1, 128), DType::F32, &device)?,
      context: Tensor::zeros((1, context_size), DType::F32, &device)?,
      device,
    })
  }

  fn process_audio_chunk(
    &mut self,
    chunk: Vec<f32>,
  ) -> Result<f32> {
    if chunk.len() < self.frame_size {
      return Ok(0.0);
    }

    let next_context = Tensor::from_slice(&chunk[self.frame_size - self.context_size..], (1, self.context_size), &self.device)?;

    let chunk_tensor = Tensor::from_vec(chunk, (1, self.frame_size), &self.device)?;
    let input = Tensor::cat(&[&self.context, &chunk_tensor], 1)?;

    let inputs: HashMap<String, Tensor> = HashMap::from_iter([("input".to_string(), input), ("sr".to_string(), self.sample_rate.clone()), ("state".to_string(), self.state.clone())]);

    let outputs = simple_eval(&self.model, inputs)?;
    let graph = self.model.graph.as_ref().unwrap();
    let out_names = &graph.output;

    let output = outputs
      .get(&out_names[0].name)
      .ok_or_else(|| anyhow::anyhow!("Missing output from model"))?
      .clone();

    self.state = outputs
      .get(&out_names[1].name)
      .ok_or_else(|| anyhow::anyhow!("Missing state output from model"))?
      .clone();

    self.context = next_context;

    let speech_prob = output.flatten_all()?.to_vec1::<f32>()?[0];
    Ok(speech_prob)
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

fn main() -> Result<()> {
  let args = Args::parse();

  let device = device(args.cpu)?;
  let model = load_model()?;
  let mut vad = VADState::new(model, i64::from(args.sample_rate), device)?;

  let input: Box<dyn std::io::Read> = if let Some(file_path) = &args.file {
    println!("Reading from file: {}", file_path.display());
    Box::new(File::open(file_path)?)
  } else {
    println!("Reading from stdin");
    Box::new(std::io::stdin().lock())
  };

  let threshold = 0.3;
  let mut speaking = false;
  let mut predictions = Vec::new();

  println!("Processing audio data... Press Ctrl+C to stop.");

  for audio_chunk_result in AudioFrameReader::new(input, vad.frame_size) {
    let audio_chunk = match audio_chunk_result {
      std::result::Result::Ok(chunk) => chunk,
      Err(e) => {
        eprintln!("Error reading audio data: {e}");
        break;
      },
    };

    if audio_chunk.len() < vad.frame_size {
      continue;
    }

    match vad.process_audio_chunk(audio_chunk) {
      std::result::Result::Ok(speech_prob) => {
        let was_speaking = speaking;
        speaking = speech_prob > threshold;

        if speaking != was_speaking {
          if speaking {
            println!("🗣️  Voice activity detected! Probability: {speech_prob:.3}");
          } else {
            println!("🔇 Voice activity stopped. Probability: {speech_prob:.3}");
          }
        }

        predictions.push(speech_prob);
        println!("VAD chunk prediction: {speech_prob:.3}");
      },
      Err(e) => {
        eprintln!("Error processing audio chunk: {e}");
      },
    }
  }

  if !predictions.is_empty() {
    #[allow(clippy::cast_precision_loss)]
    let average_prediction = predictions.iter().sum::<f32>() / predictions.len() as f32;
    println!("VAD average prediction: {average_prediction:.3}");
  }

  Ok(())
}
