use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use tokio::time::timeout;

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
