use std::{
  fs::File,
  io::Write,
  path::PathBuf,
  str::FromStr,
  time::{Duration, Instant},
};

use anyhow::Result;
use clap::Parser;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::{
  audio_manager::{AudioBuffer, AudioManager},
  vad::VADProcessor,
  whisper::{WhichWhisperModel, WhisperProcessor},
};

mod audio_manager;
mod vad;
mod whisper;

#[derive(Parser, Debug)]
struct Args {
  #[arg(long)]
  cpu:                     bool,
  #[arg(long, default_value = "16000")]
  sample_rate:             u32,
  #[arg(long)]
  device:                  Option<String>,
  #[arg(long, default_value = "0.3")]
  vad_threshold:           f32,
  #[arg(long, default_value = "500")]
  min_silence_duration_ms: u64,
  #[arg(long, default_value = "100")]
  min_speech_duration_ms:  u64,
  #[arg(long, default_value = "medium")]
  whisper_model:           Option<WhichWhisperModel>,
}

#[tokio::main]
async fn main() -> Result<()> {
  let args = Args::parse();
  println!("Parsed arguments: {args:?}");

  let device = if args.cpu {
    candle_core::Device::Cpu
  } else if candle_core::utils::cuda_is_available() {
    candle_core::Device::new_cuda(0)?
  } else if candle_core::utils::metal_is_available() {
    candle_core::Device::new_metal(0)?
  } else {
    candle_core::Device::Cpu
  };

  println!("Using device: {device:?}");

  let mut vad = VADProcessor::new(candle_core::Device::Cpu, args.vad_threshold)?;
  let mut whisper = WhisperProcessor::new(
    args
      .whisper_model
      .unwrap_or(WhichWhisperModel::Tiny),
    device.clone(),
  )?;

  let mut audio_manager = AudioManager::new(args.device.clone(), args.sample_rate)?;
  let mut audio_buffer = AudioBuffer::new(10000, args.min_speech_duration_ms, args.min_silence_duration_ms, args.sample_rate);

  let mut frame_buffer = Vec::<f32>::new();

  println!("🎤 Listening for speech... Press Ctrl+C to stop.");

  loop {
    // Try to get audio with timeout to allow for graceful shutdown
    match timeout(Duration::from_millis(100), audio_manager.receive_audio()).await {
      Ok(Ok(chunk)) => {
        process_audio_chunk(&mut vad, &mut whisper, &mut audio_buffer, &mut frame_buffer, chunk, args.sample_rate).await?;
      },
      Ok(Err(_)) => {
        println!("Audio stream ended");
        break;
      },
      Err(_) => {
        // Timeout - continue loop (allows Ctrl+C handling)
      },
    }
  }

  Ok(())
}

fn save_audio_chunk_to_file(
  chunk: &[f32],
  output_dir: PathBuf,
  sample_rate: u32,
) -> Result<PathBuf, String> {
  let timestamp = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_millis();

  let filename = format!("silero-vad-whisper-realtime_audio-chunk_{}.wav", timestamp);
  let filepath = output_dir.join(&filename);

  // Write as WAV file for easier debugging
  write_wav_file(&filepath, chunk, sample_rate).map_err(|e| format!("Failed to write audio file: {}", e))?;

  info!("Saved audio chunk to: {:?}", filepath);
  Ok(filepath)
}

fn write_wav_file(
  filepath: &PathBuf,
  samples: &[f32],
  sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
  let mut file = File::create(filepath)?;

  let num_samples = samples.len() as u32;
  let num_channels = 1u16; // Mono
  let bits_per_sample = 16u16;
  let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
  let block_align = num_channels * bits_per_sample / 8;
  let data_size = num_samples * bits_per_sample as u32 / 8;
  let file_size = 36 + data_size;

  // WAV header
  file.write_all(b"RIFF")?;
  file.write_all(&file_size.to_le_bytes())?;
  file.write_all(b"WAVE")?;

  // Format chunk
  file.write_all(b"fmt ")?;
  file.write_all(&16u32.to_le_bytes())?; // Chunk size
  file.write_all(&1u16.to_le_bytes())?; // Audio format (PCM)
  file.write_all(&num_channels.to_le_bytes())?;
  file.write_all(&sample_rate.to_le_bytes())?;
  file.write_all(&byte_rate.to_le_bytes())?;
  file.write_all(&block_align.to_le_bytes())?;
  file.write_all(&bits_per_sample.to_le_bytes())?;

  // Data chunk
  file.write_all(b"data")?;
  file.write_all(&data_size.to_le_bytes())?;

  // Convert f32 samples to i16 and write
  for sample in samples {
    let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
    file.write_all(&sample_i16.to_le_bytes())?;
  }

  Ok(())
}

async fn process_audio_chunk(
  vad: &mut VADProcessor,
  whisper: &mut WhisperProcessor,
  audio_buffer: &mut AudioBuffer,
  frame_buffer: &mut Vec<f32>, // Add this parameter
  chunk: Vec<f32>,
  sample_rate: u32,
) -> Result<()> {
  // Accumulate audio data
  frame_buffer.extend_from_slice(&chunk);

  // Process complete 512-sample frames
  while frame_buffer.len() >= 512 {
    let frame: Vec<f32> = frame_buffer.drain(..512).collect();
    let speech_prob = vad.process_chunk(&frame)?;
    let is_speech = vad.is_speech(speech_prob);

    if let Some(complete_audio) = audio_buffer.add_chunk(&frame, is_speech) {
      if let Some(ref output_dir) = PathBuf::from_str("./audio_chunks").ok() {
        if !output_dir.exists() {
          std::fs::create_dir_all(output_dir).map_err(|e| anyhow::anyhow!("Failed to create output directory: {}", e))?;
        }

        let pwd = std::env::current_dir().map_err(|e| anyhow::anyhow!("Failed to get current directory: {}", e))?;
        info!("Saving audio chunk for debugging... {}", pwd.display());
        let sr = 16000; // Default sample rate
        match save_audio_chunk_to_file(complete_audio.clone().as_slice(), pwd.join(output_dir), sr) {
          Ok(filepath) => {
            info!("Audio chunk saved for debugging: {:?}", filepath);
          },
          Err(e) => {
            warn!("Failed to save audio chunk for debugging: {}", e);
          },
        }
      }

      transcribe_audio(whisper, complete_audio, sample_rate).await?;
    }
  }

  Ok(())
}

#[allow(clippy::unused_async)]
async fn transcribe_audio(
  whisper: &mut WhisperProcessor,
  audio: Vec<f32>,
  sample_rate: u32,
) -> Result<()> {
  // Calculate duration of the audio in seconds

  #[allow(clippy::cast_precision_loss)]
  let duration_secs = audio.len() as f32 / sample_rate as f32;
  println!("🔄 Transcribing {duration_secs:.2}s audio...");

  let start_time = Instant::now();
  let transcript = whisper.transcribe(&audio)?;
  let duration = start_time.elapsed();

  if transcript.trim().is_empty() || transcript.contains("[BLANK_AUDIO]") {
    println!("   (No speech detected)");
    return Ok(());
  }

  println!("📝 Transcript ({:.2}s): \"{}\"", duration.as_secs_f32(), transcript.trim());
  Ok(())
}
