use std::time::Instant;

use anyhow::Result;
use cpal::{
  InputCallbackInfo,
  traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};

pub struct AudioManager {
  _stream:      cpal::Stream,
  audio_rx:     crossbeam_channel::Receiver<Vec<f32>>,
  resampler:    Option<FastFixedIn<f32>>,
  buffered_pcm: Vec<f32>,
}

impl AudioManager {
  pub fn new(
    device_name: Option<String>,
    target_sample_rate: u32,
  ) -> Result<Self> {
    let host = cpal::default_host();
    let device = match device_name {
      None => host.default_input_device(),
      Some(name) => host
        .input_devices()?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false)),
    }
    .ok_or_else(|| {
      anyhow::anyhow!(
        "No input device found, current available devices: {:?}",
        host
          .input_devices()
          .unwrap()
          .map(|d| d
            .name()
            .unwrap_or_else(|_| "Unnamed Device".to_string()))
          .collect::<Vec<String>>()
          .join(", ")
      )
    })?;

    println!("Using audio input device: {}", device.name()?);

    let config = device.default_input_config()?;
    let channel_count = config.channels() as usize;
    let device_sample_rate = config.sample_rate().0;

    println!("Device sample rate: {device_sample_rate}Hz, Target: {target_sample_rate}Hz");

    let (tx, rx) = crossbeam_channel::unbounded();

    let stream = device.build_input_stream(
      &config.into(),
      move |data: &[f32], _: &InputCallbackInfo| {
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
      |err| eprintln!("Audio stream error: {err}"),
      None,
    )?;

    stream.play()?;

    let resampler = if device_sample_rate == target_sample_rate {
      None
    } else {
      let resample_ratio = f64::from(target_sample_rate) / f64::from(device_sample_rate);

      Some(FastFixedIn::new(
        resample_ratio,
        10.0, // max_resample_ratio_relative
        PolynomialDegree::Septic,
        1024, // chunk_size
        1,    // channels
      )?)
    };

    Ok(Self { _stream: stream, audio_rx: rx, resampler, buffered_pcm: Vec::new() })
  }

  #[allow(clippy::future_not_send, clippy::unused_async)]
  pub async fn receive_audio(&mut self) -> Result<Vec<f32>> {
    let chunk = self.audio_rx.recv()?;

    if let Some(ref mut resampler) = self.resampler {
      // Need to resample
      self.buffered_pcm.extend_from_slice(&chunk);

      let mut resampled_audio = Vec::new();

      // Process in chunks of 1024 samples
      let full_chunks = self.buffered_pcm.len() / 1024;
      let remainder = self.buffered_pcm.len() % 1024;

      for chunk_idx in 0..full_chunks {
        let chunk_slice = &self.buffered_pcm[chunk_idx * 1024..(chunk_idx + 1) * 1024];
        let resampled = resampler.process(&[chunk_slice], None)?;
        resampled_audio.extend_from_slice(&resampled[0]);
      }

      // Handle remainder
      if remainder == 0 {
        self.buffered_pcm.clear();
      } else {
        self
          .buffered_pcm
          .copy_within(full_chunks * 1024.., 0);
        self.buffered_pcm.truncate(remainder);
      }

      Ok(resampled_audio)
    } else {
      // No resampling needed, return the chunk directly
      Ok(chunk)
    }
  }
}

pub struct AudioBuffer {
  buffer:                       Vec<f32>,
  max_duration_samples:         usize,
  min_speech_duration_samples:  usize,
  min_silence_duration_samples: usize,
  is_recording:                 bool,
  silence_start:                Option<Instant>,
  speech_start:                 Option<Instant>,
  samples_since_speech_start:   usize,
  samples_since_silence_start:  usize,
  sample_rate:                  usize,
}

impl AudioBuffer {
  pub const fn new(
    max_duration_ms: u64,
    min_speech_duration_ms: u64,
    min_silence_duration_ms: u64,
    sample_rate: u32,
  ) -> Self {
    let sample_rate = sample_rate as usize;
    Self {
      buffer: Vec::new(),
      max_duration_samples: (max_duration_ms * sample_rate as u64 / 1000) as usize,
      min_speech_duration_samples: (min_speech_duration_ms * sample_rate as u64 / 1000) as usize,
      min_silence_duration_samples: (min_silence_duration_ms * sample_rate as u64 / 1000) as usize,
      is_recording: false,
      silence_start: None,
      speech_start: None,
      samples_since_speech_start: 0,
      samples_since_silence_start: 0,
      sample_rate,
    }
  }

  pub fn add_chunk(
    &mut self,
    chunk: &[f32],
    is_speech: bool,
  ) -> Option<Vec<f32>> {
    if is_speech {
      #[allow(clippy::if_not_else)]
      if !self.is_recording {
        if self.speech_start.is_none() {
          self.speech_start = Some(Instant::now());
          self.samples_since_speech_start = 0;
        }

        self.samples_since_speech_start += chunk.len();

        if self.samples_since_speech_start >= self.min_speech_duration_samples {
          self.is_recording = true;
          self.silence_start = None;
          self.samples_since_silence_start = 0;
          println!("🚀 Started recording");
        }
      } else {
        // Reset silence tracking
        self.silence_start = None;
        self.samples_since_silence_start = 0;
      }
    } else {
      // Reset speech tracking
      self.speech_start = None;
      self.samples_since_speech_start = 0;

      if self.is_recording {
        if self.silence_start.is_none() {
          self.silence_start = Some(Instant::now());
          self.samples_since_silence_start = 0;
        }

        self.samples_since_silence_start += chunk.len();

        if self.samples_since_silence_start >= self.min_silence_duration_samples {
          // End of speech detected
          if !self.buffer.is_empty() {
            let result = self.buffer.clone();
            self.reset();
            #[allow(clippy::cast_precision_loss)]
            let duration_secs = result.len() as f32 / self.sample_rate as f32;
            println!("🔇 Stopped recording, {duration_secs:.2}s");
            return Some(result);
          }

          self.reset();
        }
      }
    }

    if self.is_recording {
      self.buffer.extend_from_slice(chunk);

      // Check if buffer exceeds max duration
      if self.buffer.len() >= self.max_duration_samples {
        let result = self.buffer.clone();
        self.reset();
        println!("⏰ Max duration reached, {} samples", result.len());
        return Some(result);
      }
    }

    None
  }

  fn reset(&mut self) {
    self.buffer.clear();
    self.is_recording = false;
    self.silence_start = None;
    self.speech_start = None;
    self.samples_since_speech_start = 0;
    self.samples_since_silence_start = 0;
  }
}
