use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use tokio::time::timeout;

// Import the new pipelines and necessary configs
use crate::{
  audio_manager::{AudioBuffer, AudioManager},
  vad::SileroVADPipeline,
  whisper::{GenerationConfig, WhichModel, WhisperPipeline},
};

mod audio_manager;
mod vad;
mod whisper;
mod whisper_processor; // Make sure to declare the new module

#[derive(Parser, Debug)]
struct Args {
  #[arg(long, default_value = "16000")]
  sample_rate:             u32,
  #[arg(long)]
  device:                  Option<String>,
  #[arg(long, default_value = "0.5")]
  vad_threshold:           f32,
  #[arg(long, default_value = "500")]
  min_silence_duration_ms: u64,
  #[arg(long, default_value = "100")]
  min_speech_duration_ms:  u64,
  #[arg(long, default_value = "base")]
  whisper_model:           Option<WhichModel>,
}

#[tokio::main]
async fn main() -> Result<()> {
  let args = Args::parse();
  println!("Parsed arguments: {args:?}");

  // 1. Initialize the Silero VAD Pipeline
  let mut vad = SileroVADPipeline::new(args.sample_rate, args.vad_threshold)?;

  // 2. Initialize the LiteWhisper Pipeline
  let (model_name, revision) = args.whisper_model.unwrap().model_and_revision();
  let mut whisper = WhisperPipeline::new(args.whisper_model.unwrap(), model_name, revision)?;

  // 3. Set up audio input and buffering
  let mut audio_manager = AudioManager::new(args.device.clone(), args.sample_rate)?;
  // Use a 30-second buffer, as that's what Whisper works with
  let mut audio_buffer = AudioBuffer::new(30000, args.min_speech_duration_ms, args.min_silence_duration_ms, args.sample_rate);

  let mut frame_buffer = Vec::<f32>::new();

  println!("🎤 Listening for speech... Press Ctrl+C to stop.");

  loop {
    match timeout(Duration::from_millis(100), audio_manager.receive_audio()).await {
      Ok(Ok(chunk)) => {
        process_audio_chunk(&mut vad, &mut whisper, &mut audio_buffer, &mut frame_buffer, chunk, args.sample_rate).await?;
      },
      Ok(Err(_)) => {
        println!("Audio stream ended");
        break;
      },
      Err(_) => {
        // Timeout allows for graceful shutdown via Ctrl+C
      },
    }
  }

  Ok(())
}

async fn process_audio_chunk(
  vad: &mut SileroVADPipeline,
  whisper: &mut WhisperPipeline,
  audio_buffer: &mut AudioBuffer,
  frame_buffer: &mut Vec<f32>,
  chunk: Vec<f32>,
  sample_rate: u32,
) -> Result<()> {
  frame_buffer.extend_from_slice(&chunk);

  // Silero VAD works with specific frame sizes. 512 is a good choice for 16kHz audio.
  const VAD_FRAME_SIZE: usize = 512;

  while frame_buffer.len() >= VAD_FRAME_SIZE {
    let frame: Vec<f32> = frame_buffer.drain(..VAD_FRAME_SIZE).collect();

    // Use the VAD pipeline to get speech probability
    let speech_prob = vad.process_chunk(&frame)?;
    let is_speech = vad.is_speech(speech_prob);

    // Add the frame to the audio buffer and check if a full utterance is ready
    if let Some(complete_audio) = audio_buffer.add_chunk(&frame, is_speech) {
      if !complete_audio.is_empty() {
        // For debugging, you can save the audio chunk
        // ... (your save_audio_chunk_to_file logic can go here)

        // Transcribe the complete audio utterance
        transcribe_audio(whisper, complete_audio, sample_rate).await?;
      }
    }
  }

  Ok(())
}

async fn transcribe_audio(
  whisper: &mut WhisperPipeline,
  audio: Vec<f32>,
  sample_rate: u32,
) -> Result<()> {
  #[allow(clippy::cast_precision_loss)]
  let duration_secs = audio.len() as f32 / sample_rate as f32;
  println!("🔄 Transcribing {duration_secs:.2}s audio...");

  // Configure the transcription task
  let gen_config = GenerationConfig::default();

  let start_time = Instant::now();
  let transcript = whisper.transcribe(&audio, &gen_config)?;
  let duration = start_time.elapsed();

  // Whisper models often output a specific token for silence or non-speech.
  if transcript.trim().is_empty() || transcript.contains("<|nospeech|>") {
    println!("   (No speech detected)");
    return Ok(());
  }

  println!("📝 Transcript ({:.2}s): \"{}\"", duration.as_secs_f32(), transcript.trim());
  Ok(())
}

// ... (Your save_audio_chunk_to_file and write_wav_file functions remain the same)
