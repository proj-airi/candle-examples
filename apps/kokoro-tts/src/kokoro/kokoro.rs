use std::{
  collections::HashMap,
  fs::File,
  path::Path,
  sync::{Arc, Mutex},
};

use espeak_rs::text_to_phonemes;
use ndarray::Array3;
use ndarray_npy::NpzReader;

use crate::kokoro::tokenize::tokenize;

#[derive(Debug, Clone)]
pub struct TTSOpts<'a> {
  pub txt:             &'a str,
  pub lan:             &'a str,
  pub style_name:      &'a str,
  pub save_path:       &'a str,
  pub mono:            bool,
  pub speed:           f32,
  pub initial_silence: Option<usize>,
}

#[derive(Clone)]
pub struct TTSKoko {
  #[allow(dead_code)]
  model_path:  String,
  model:       Arc<Mutex<crate::onnx::kokoro::OrtKoko>>,
  styles:      HashMap<String, Vec<[[f32; 256]; 1]>>,
  init_config: InitConfig,
}

#[derive(Clone)]
pub struct InitConfig {
  pub model_url:   String,
  pub voices_url:  String,
  pub sample_rate: u32,
}

impl Default for InitConfig {
  fn default() -> Self {
    Self {
      model_url:   "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx".into(),
      voices_url:  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin".into(),
      sample_rate: 24000,
    }
  }
}

impl TTSKoko {
  pub async fn new(
    model_path: &str,
    voices_path: &str,
  ) -> Self {
    Self::from_config(model_path, voices_path, InitConfig::default()).await
  }

  pub async fn from_config(
    model_path: &str,
    voices_path: &str,
    cfg: InitConfig,
  ) -> Self {
    if !Path::new(model_path).exists() {
      crate::utils::fileio::download_file_from_url(cfg.model_url.as_str(), model_path)
        .await
        .expect("download model failed.");
    }

    if !Path::new(voices_path).exists() {
      crate::utils::fileio::download_file_from_url(cfg.voices_url.as_str(), voices_path)
        .await
        .expect("download voices data file failed.");
    }

    let model = Arc::new(Mutex::new(crate::onnx::kokoro::OrtKoko::new(model_path.to_string()).expect("Failed to create Kokoro TTS model")));
    // TODO: if(not streaming) { model.print_info(); }
    // model.print_info();

    let styles = Self::load_voices(voices_path);

    Self { model_path: model_path.to_string(), model, styles, init_config: cfg }
  }

  fn split_text_into_chunks(
    &self,
    text: &str,
    max_tokens: usize,
  ) -> Vec<String> {
    let mut chunks = Vec::new();

    // First split by sentences - using common sentence ending punctuation
    let sentences: Vec<&str> = text
      .split(['.', '?', '!', ';'])
      .filter(|s| !s.trim().is_empty())
      .collect();

    let mut current_chunk = String::new();

    for sentence in sentences {
      // Clean up the sentence and add back punctuation
      let sentence = format!("{}.", sentence.trim());

      // Convert to phonemes to check token count
      let sentence_phonemes = text_to_phonemes(&sentence, "en", None, true, false)
        .unwrap_or_default()
        .join("");
      let token_count = tokenize(&sentence_phonemes).len();

      if token_count > max_tokens {
        // If single sentence is too long, split by words
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut word_chunk = String::new();

        for word in words {
          let test_chunk = if word_chunk.is_empty() {
            word.to_string()
          } else {
            format!("{word_chunk} {word}")
          };

          let test_phonemes = text_to_phonemes(&test_chunk, "en", None, true, false)
            .unwrap_or_default()
            .join("");
          let test_tokens = tokenize(&test_phonemes).len();

          if test_tokens > max_tokens {
            if !word_chunk.is_empty() {
              chunks.push(word_chunk);
            }
            word_chunk = word.to_string();
          } else {
            word_chunk = test_chunk;
          }
        }

        if !word_chunk.is_empty() {
          chunks.push(word_chunk);
        }
      } else if !current_chunk.is_empty() {
        // Try to append to current chunk
        let test_text = format!("{current_chunk} {sentence}");
        let test_phonemes = text_to_phonemes(&test_text, "en", None, true, false)
          .unwrap_or_default()
          .join("");
        let test_tokens = tokenize(&test_phonemes).len();

        if test_tokens > max_tokens {
          // If combining would exceed limit, start new chunk
          chunks.push(current_chunk);
          current_chunk = sentence;
        } else {
          current_chunk = test_text;
        }
      } else {
        current_chunk = sentence;
      }
    }

    // Add the last chunk if not empty
    if !current_chunk.is_empty() {
      chunks.push(current_chunk);
    }

    chunks
  }

  pub fn tts_raw_audio(
    &self,
    txt: &str,
    lan: &str,
    style_name: &str,
    speed: f32,
    initial_silence: Option<usize>,
  ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Split text into appropriate chunks
    let chunks = self.split_text_into_chunks(txt, 500); // Using 500 to leave 12 tokens of margin
    let mut final_audio = Vec::new();

    for chunk in chunks {
      // Convert chunk to phonemes
      let phonemes = text_to_phonemes(&chunk, lan, None, true, false)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?
        .join("");
      eprintln!("phonemes: {phonemes}");
      let mut tokens = tokenize(&phonemes);

      for _ in 0..initial_silence.unwrap_or(0) {
        tokens.insert(0, 30);
      }

      // Get style vectors once
      let styles = self.mix_styles(style_name, tokens.len())?;

      // pad a 0 to start and end of tokens
      let mut padded_tokens = vec![0];
      for &token in &tokens {
        padded_tokens.push(token);
      }
      padded_tokens.push(0);

      let tokens = vec![padded_tokens];

      match self
        .model
        .lock()
        .unwrap()
        .infer(tokens, styles.clone(), speed)
      {
        Ok(chunk_audio) => {
          let chunk_audio: Vec<f32> = chunk_audio.iter().copied().collect();
          final_audio.extend_from_slice(&chunk_audio);
        },
        Err(e) => {
          eprintln!("Error processing chunk: {e:?}");
          eprintln!("Chunk text was: {chunk:?}");
          return Err(Box::new(std::io::Error::other(format!("Chunk processing failed: {e:?}"))));
        },
      }
    }

    Ok(final_audio)
  }

  pub fn tts(
    &self,
    TTSOpts { txt, lan, style_name, save_path, mono, speed, initial_silence }: TTSOpts,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let audio = self.tts_raw_audio(txt, lan, style_name, speed, initial_silence)?;

    // Save to file
    if mono {
      let spec = hound::WavSpec {
        channels:        1,
        sample_rate:     self.init_config.sample_rate,
        bits_per_sample: 32,
        sample_format:   hound::SampleFormat::Float,
      };

      let mut writer = hound::WavWriter::create(save_path, spec)?;
      for &sample in &audio {
        writer.write_sample(sample)?;
      }
      writer.finalize()?;
    } else {
      let spec = hound::WavSpec {
        channels:        2,
        sample_rate:     self.init_config.sample_rate,
        bits_per_sample: 32,
        sample_format:   hound::SampleFormat::Float,
      };

      let mut writer = hound::WavWriter::create(save_path, spec)?;
      for &sample in &audio {
        writer.write_sample(sample)?;
        writer.write_sample(sample)?;
      }
      writer.finalize()?;
    }
    eprintln!("Audio saved to {save_path}");
    Ok(())
  }

  pub fn mix_styles(
    &self,
    style_name: &str,
    tokens_len: usize,
  ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    if !style_name.contains("+") {
      #[allow(clippy::option_if_let_else)]
      if let Some(style) = self.styles.get(style_name) {
        let styles = vec![style[tokens_len][0].to_vec()];
        Ok(styles)
      } else {
        Err(format!("can not found from styles_map: {style_name}").into())
      }
    } else {
      eprintln!("parsing style mix");
      let styles: Vec<&str> = style_name.split('+').collect();

      let mut style_names = Vec::new();
      let mut style_portions = Vec::new();

      for style in styles {
        if let Some((name, portion)) = style.split_once('.')
          && let Ok(portion) = portion.parse::<f32>()
        {
          style_names.push(name);
          style_portions.push(portion * 0.1);
        }
      }
      eprintln!("styles: {style_names:?}, portions: {style_portions:?}");

      let mut blended_style = vec![vec![0.0; 256]; 1];

      for (name, portion) in style_names.iter().zip(style_portions.iter()) {
        if let Some(style) = self.styles.get(*name) {
          let style_slice = &style[tokens_len][0]; // This is a [256] array
          // Blend into the blended_style
          #[allow(clippy::needless_range_loop)]
          for j in 0..256 {
            blended_style[0][j] += style_slice[j] * portion;
          }
        }
      }
      Ok(blended_style)
    }
  }

  fn load_voices(voices_path: &str) -> HashMap<String, Vec<[[f32; 256]; 1]>> {
    let mut npz = NpzReader::new(File::open(voices_path).unwrap()).unwrap();
    let mut map = HashMap::new();

    for voice in npz.names().unwrap() {
      let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
      let voice_data = voice_data.unwrap();
      let mut tensor = vec![[[0.0; 256]; 1]; 511];
      for (i, inner_value) in voice_data.outer_iter().enumerate() {
        for (j, inner_inner_value) in inner_value.outer_iter().enumerate() {
          for (k, number) in inner_inner_value.iter().enumerate() {
            tensor[i][j][k] = *number;
          }
        }
      }
      map.insert(voice, tensor);
    }

    let sorted_voices = {
      let mut voices = map.keys().collect::<Vec<_>>();
      voices.sort();
      voices
    };

    eprintln!("voice styles loaded: {sorted_voices:?}");
    map
  }
}
