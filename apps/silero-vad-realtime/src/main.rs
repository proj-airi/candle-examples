use std::{
  collections::HashMap,
  sync::mpsc,
  time::{Duration, Instant},
};

use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};
use candle_onnx::simple_eval;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::Resampler;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Run on CPU or GPU
  #[arg(long)]
  cpu:                     bool,
  /// Enable tracing (generates a trace-timestamp.json file).
  #[arg(long)]
  tracing:                 bool,
  /// Input device to use (e.g., "hw:0,0" for ALSA or "default" for `PulseAudio`).
  #[arg(long)]
  device:                  Option<String>,
  /// Sample rate for the audio input (VAD model expects 16000 or 8000).
  #[arg(long, default_value = "16000")]
  sample_rate:             u32,
  /// Minimum duration (in milliseconds) below threshold before considering speech stopped
  #[arg(long, default_value = "500")]
  min_silence_duration_ms: u64,
  /// Minimum duration (in milliseconds) above threshold before considering speech started
  #[arg(long, default_value = "100")]
  min_speech_duration_ms:  u64,
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

// Return both the stream and the device sample rate for resampling setup
fn setup_audio_capture(args: &Args) -> Result<(cpal::Stream, mpsc::Receiver<Vec<f32>>, u32)> {
  let host = cpal::default_host();
  let device = match &args.device {
    None => host.default_input_device(),
    Some(device_name) => host.input_devices()?.find(|d| {
      d.name()
        .map(|n| n == *device_name)
        .unwrap_or(false)
    }),
  }
  .ok_or_else(|| anyhow::anyhow!("Failed to find input device"))?;

  println!("Using input device: {}", device.name()?);

  let config = device.default_input_config()?;
  println!("Device config: {config:?}");

  let channel_count = config.channels() as usize;
  let device_sample_rate = config.sample_rate().0;

  let (tx, rx) = mpsc::channel();

  let stream = device.build_input_stream(
    &config.into(),
    move |data: &[f32], _: &cpal::InputCallbackInfo| {
      // Extract mono audio (first channel only)
      let mono_data = data
        .iter()
        .step_by(channel_count)
        .copied()
        .collect::<Vec<f32>>();

      if !mono_data.is_empty() {
        let _ = tx.send(mono_data);
      }
    },
    |err| eprintln!("Error in audio stream: {err}"),
    None,
  )?;

  stream.play()?;
  Ok((stream, rx, device_sample_rate))
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

struct SpeechStateManager {
  is_speaking:          bool,
  threshold:            f32,
  min_silence_duration: Duration,
  min_speech_duration:  Duration,

  above_threshold_since: Option<Instant>,
  below_threshold_since: Option<Instant>,
  state_change_time:     Instant,
}

impl SpeechStateManager {
  fn new(
    threshold: f32,
    min_silence_duration_ms: u64,
    min_speech_duration_ms: u64,
  ) -> Self {
    Self {
      is_speaking: false,
      threshold,
      min_silence_duration: Duration::from_millis(min_silence_duration_ms),
      min_speech_duration: Duration::from_millis(min_speech_duration_ms),
      above_threshold_since: None,
      below_threshold_since: None,
      state_change_time: Instant::now(),
    }
  }

  fn update(
    &mut self,
    speech_prob: f32,
  ) -> bool {
    let now = Instant::now();
    let above_threshold = speech_prob > self.threshold;

    // Track when we FIRST go above/below threshold (not every update)
    if above_threshold && self.above_threshold_since.is_none() {
      // Just crossed above threshold
      self.above_threshold_since = Some(now);
      self.below_threshold_since = None; // Reset below counter
      println!("🔊 Crossed above threshold at {speech_prob:.3}");
    } else if !above_threshold && self.below_threshold_since.is_none() {
      // Just crossed below threshold
      self.below_threshold_since = Some(now);
      self.above_threshold_since = None; // Reset above counter
      println!("🔉 Crossed below threshold at {speech_prob:.3}");
    }

    let was_speaking = self.is_speaking;

    // State transition logic with proper timing
    if !self.is_speaking && above_threshold {
      // Not speaking -> potentially speaking
      if let Some(first_above) = self.above_threshold_since {
        let duration_above = now.duration_since(first_above);
        println!("Above threshold for {:.1}ms (need {:.1}ms)", duration_above.as_millis(), self.min_speech_duration.as_millis());

        if duration_above >= self.min_speech_duration {
          self.is_speaking = true;
          self.state_change_time = now;
          println!("✅ State changed to SPEAKING");
        }
      }
    } else if self.is_speaking && !above_threshold {
      // Speaking -> potentially not speaking
      if let Some(first_below) = self.below_threshold_since {
        let duration_below = now.duration_since(first_below);
        println!("Below threshold for {:.1}ms (need {:.1}ms)", duration_below.as_millis(), self.min_silence_duration.as_millis());

        if duration_below >= self.min_silence_duration {
          self.is_speaking = false;
          self.state_change_time = now;
          println!("✅ State changed to NOT SPEAKING");
        }
      }
    }

    // Return true if state changed
    was_speaking != self.is_speaking
  }

  const fn is_speaking(&self) -> bool {
    self.is_speaking
  }

  fn time_since_state_change(&self) -> Duration {
    self.state_change_time.elapsed()
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

  // Get the device sample rate from setup_audio_capture
  let (_stream, audio_rx, device_sample_rate) = setup_audio_capture(&args)?;
  let mut vad = VADState::new(model, i64::from(args.sample_rate), device)?;

  let target_sample_rate = args.sample_rate;
  let resample_ratio = f64::from(target_sample_rate) / f64::from(device_sample_rate);

  println!("Device sample rate: {device_sample_rate}Hz, Target: {target_sample_rate}Hz, Ratio: {resample_ratio:.4}");

  // Create the resampler properly
  let mut resampler = rubato::FastFixedIn::new(
    resample_ratio,
    10.0, // max_resample_ratio_relative
    rubato::PolynomialDegree::Septic,
    1024, // chunk_size
    1,    // channels
  )?;

  let mut buffered_pcm = Vec::new();
  let mut audio_buffer = Vec::new();

  // Create speech state manager with configurable timing
  let mut speech_state = SpeechStateManager::new(
    0.3, // threshold
    args.min_silence_duration_ms,
    args.min_speech_duration_ms,
  );

  println!("Listening for voice activity... Press Ctrl+C to stop.");
  println!("Min silence duration: {}ms, Min speech duration: {}ms", args.min_silence_duration_ms, args.min_speech_duration_ms);

  loop {
    let chunk = match audio_rx.recv_timeout(Duration::from_millis(100)) {
      std::result::Result::Ok(c) => {
        if c.is_empty() {
          continue;
        }
        c
      },
      std::result::Result::Err(mpsc::RecvTimeoutError::Timeout) => {
        continue;
      },
      std::result::Result::Err(mpsc::RecvTimeoutError::Disconnected) => {
        println!("Audio stream ended.");
        break;
      },
    };

    // Resample audio following the Whisper pattern
    buffered_pcm.extend_from_slice(&chunk);

    // Process in chunks of 1024 samples
    let full_chunks = buffered_pcm.len() / 1024;
    let remainder = buffered_pcm.len() % 1024;

    for chunk_idx in 0..full_chunks {
      let chunk_slice = &buffered_pcm[chunk_idx * 1024..(chunk_idx + 1) * 1024];
      let resampler = resampler.process(&[chunk_slice], None)?;
      audio_buffer.extend_from_slice(&resampler[0]);
    }

    // Handle remainder (following Whisper pattern exactly)
    if remainder == 0 {
      buffered_pcm.clear();
    } else {
      buffered_pcm.copy_within(full_chunks * 1024.., 0);
      buffered_pcm.truncate(remainder);
    }

    // Process VAD frames
    while audio_buffer.len() >= vad.frame_size {
      let frame = audio_buffer
        .drain(..vad.frame_size)
        .collect::<Vec<f32>>();

      match vad.process_audio_chunk(frame) {
        std::result::Result::Ok(speech_prob) => {
          let state_changed = speech_state.update(speech_prob);

          if state_changed {
            if speech_state.is_speaking() {
              println!("🗣️  Voice activity detected! Probability: {speech_prob:.3}");
            } else {
              println!(
                "🔇 Voice activity stopped. Probability: {speech_prob:.3} (was speaking for {:.1}s)",
                speech_state
                  .time_since_state_change()
                  .as_secs_f32()
              );
            }
          }
        },
        Err(e) => {
          eprintln!("Error processing audio chunk: {e}");
        },
      }
    }
  }

  Ok(())
}
