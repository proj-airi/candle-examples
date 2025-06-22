use std::time::Instant;

pub struct AudioBuffer {
  buffer: Vec<f32>,
  max_duration_samples: usize,
  min_speech_duration_samples: usize,
  min_silence_duration_samples: usize,
  is_recording: bool,
  silence_start: Option<Instant>,
  speech_start: Option<Instant>,
  samples_since_speech_start: usize,
  samples_since_silence_start: usize,
  sample_rate: usize,
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
