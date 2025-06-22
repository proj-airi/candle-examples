use std::{collections::HashMap, sync::Arc, time::{Duration, Instant}};

use anyhow::Result;
use axum::{
  Json,
  extract::{Multipart, State},
  http::StatusCode,
  response::{
    IntoResponse,
    Response,
    sse::{Event, KeepAlive, Sse},
  },
};
use futures::stream::{self, Stream};
use symphonia::{
  core::{
    audio::{AudioBufferRef, Signal},
    codecs::DecoderOptions,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
  },
  default::get_probe,
};

use crate::{
  AppState,
  api::{ErrorDetail, ErrorResponse, StreamChunk, TranscriptionResponse},
  audio_manager::AudioBuffer,
};

// Performance statistics struct
#[derive(Debug)]
struct ProcessingStats {
  total_duration:                 Duration,
  audio_conversion_duration:      Duration,
  model_loading_duration:         Duration,
  vad_processing_duration:        Duration,
  whisper_transcription_duration: Duration,
  audio_length_seconds:           f32,
}

impl ProcessingStats {
  fn new() -> Self {
    Self {
      total_duration:                 Duration::ZERO,
      audio_conversion_duration:      Duration::ZERO,
      model_loading_duration:         Duration::ZERO,
      vad_processing_duration:        Duration::ZERO,
      whisper_transcription_duration: Duration::ZERO,
      audio_length_seconds:           0.0,
    }
  }

  fn print_summary(&self) {
    println!("📊 Processing Statistics:");
    println!("  📁 Audio conversion: {:.2}ms", self.audio_conversion_duration.as_secs_f64() * 1000.0);
    println!("  🧠 Model loading: {:.2}ms", self.model_loading_duration.as_secs_f64() * 1000.0);
    println!("  🎯 VAD processing: {:.2}ms", self.vad_processing_duration.as_secs_f64() * 1000.0);
    println!("  🗣️  Whisper transcription: {:.2}ms", self.whisper_transcription_duration.as_secs_f64() * 1000.0);
    println!("  ⏱️  Total processing: {:.2}ms", self.total_duration.as_secs_f64() * 1000.0);
    println!("  🎵 Audio length: {:.2}s", self.audio_length_seconds);
    if self.audio_length_seconds > 0.0 {
      let real_time_factor = self.total_duration.as_secs_f64() / self.audio_length_seconds as f64;
      println!("  ⚡ Real-time factor: {:.2}x", real_time_factor);
    }
  }
}

pub async fn transcribe_audio(
  State(state): State<Arc<AppState>>,
  mut multipart: Multipart,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
  let start_time = Instant::now();
  let mut stats = ProcessingStats::new();

  // Extract both audio file and parameters from multipart form
  let (audio_data, params) = match extract_multipart_data(&mut multipart).await {
    Ok(data) => data,
    Err(e) => {
      return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: ErrorDetail {
            message:    format!("Failed to extract form data: {}", e),
            error_type: "invalid_request_error".to_string(),
            param:      Some("form".to_string()),
            code:       None,
          },
        }),
      ));
    },
  };

  println!("Request params: {:?}", params);

  // Parse streaming parameter from form data
  let stream_enabled = params
    .get("stream")
    .map(|s| s.parse::<bool>().unwrap_or(false))
    .unwrap_or(false);

  // Get model name from parameters and clone it to make it owned
  let model_name = params
    .get("model")
    .map(|s| s.clone()) // Clone to make it owned
    .unwrap_or_else(|| "tiny".to_string()); // Use tiny as default

  println!("Using model: {}, streaming: {}", model_name, stream_enabled);

  // Convert audio to PCM format with timing
  let conversion_start = Instant::now();
  let pcm_data = match convert_audio_to_pcm(&audio_data).await {
    Ok(data) => data,
    Err(e) => {
      return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: ErrorDetail {
            message:    format!("Failed to process audio file: {}", e),
            error_type: "invalid_request_error".to_string(),
            param:      Some("file".to_string()),
            code:       None,
          },
        }),
      ));
    },
  };
  stats.audio_conversion_duration = conversion_start.elapsed();
  stats.audio_length_seconds = pcm_data.len() as f32 / 16000.0; // Assuming 16kHz sample rate

  println!("Audio data length: {} samples ({:.2}s)", pcm_data.len(), stats.audio_length_seconds);

  if stream_enabled {
    // Return streaming response
    let stream = create_transcription_stream(state, model_name, pcm_data, stats).await?;
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok(sse.into_response())
  } else {
    // Return single response
    match transcribe_audio_complete(state, model_name, pcm_data, &mut stats).await {
      Ok(transcript) => {
        stats.total_duration = start_time.elapsed();
        stats.print_summary();
        Ok(Json(TranscriptionResponse { text: transcript }).into_response())
      },
      Err(e) => {
        stats.total_duration = start_time.elapsed();
        stats.print_summary();
        Err((
          StatusCode::INTERNAL_SERVER_ERROR,
          Json(ErrorResponse {
            error: ErrorDetail {
              message:    format!("Transcription failed: {}", e),
              error_type: "server_error".to_string(),
              param:      None,
              code:       None,
            },
          }),
        ))
      },
    }
  }
}

// Extract both audio file and parameters from multipart form data
async fn extract_multipart_data(multipart: &mut Multipart) -> Result<(Vec<u8>, HashMap<String, String>)> {
  let mut audio_data = None;
  let mut params = HashMap::new();

  while let Some(field) = multipart.next_field().await? {
    if let Some(name) = field.name() {
      let name = name.to_string(); // Clone the name first to avoid borrow conflict
      if name == "file" {
        // Extract audio file
        let data = field.bytes().await?;
        audio_data = Some(data.to_vec());
      } else {
        // Extract form parameters
        let value = field.text().await?;
        params.insert(name, value);
      }
    }
  }

  let audio = audio_data.ok_or_else(|| anyhow::anyhow!("No file field found in multipart data"))?;
  Ok((audio, params))
}

// Convert various audio formats to PCM
async fn convert_audio_to_pcm(audio_data: &[u8]) -> Result<Vec<f32>> {
  let cursor = std::io::Cursor::new(audio_data.to_vec());
  let media_source = MediaSourceStream::new(Box::new(cursor), Default::default());

  let mut hint = Hint::new();
  hint.mime_type("audio/wav"); // You might want to detect this automatically

  let meta_opts: MetadataOptions = Default::default();
  let fmt_opts: FormatOptions = Default::default();

  let probed = get_probe().format(&hint, media_source, &fmt_opts, &meta_opts)?;

  let mut format = probed.format;
  let track = format
    .tracks()
    .iter()
    .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
    .ok_or_else(|| anyhow::anyhow!("No audio track found"))?;

  let dec_opts: DecoderOptions = Default::default();
  let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

  let track_id = track.id;
  let mut pcm_data = Vec::new();

  // Decode the audio
  while let Ok(packet) = format.next_packet() {
    if packet.track_id() != track_id {
      continue;
    }

    match decoder.decode(&packet)? {
      AudioBufferRef::F32(buf) => {
        for &sample in buf.chan(0) {
          pcm_data.push(sample);
        }
      },
      AudioBufferRef::S16(buf) => {
        for &sample in buf.chan(0) {
          pcm_data.push(f32::from(sample) / f32::from(i16::MAX));
        }
      },
      AudioBufferRef::S32(buf) => {
        for &sample in buf.chan(0) {
          pcm_data.push(sample as f32 / i32::MAX as f32);
        }
      },
      _ => {
        anyhow::bail!("Unsupported audio format");
      },
    }
  }

  Ok(pcm_data)
}

// Process complete audio file and return full transcript
async fn transcribe_audio_complete(
  state: Arc<AppState>,
  model_name: String, // Change to owned String
  audio_data: Vec<f32>,
  stats: &mut ProcessingStats,
) -> Result<String> {
  let sample_rate = 16000;

  // Get the appropriate Whisper processor for this model with timing
  let model_loading_start = Instant::now();
  let whisper_processor = state.get_whisper_processor(&model_name).await?;
  stats.model_loading_duration = model_loading_start.elapsed();

  // Process audio through VAD and Whisper
  let mut vad = state.vad.lock().await;
  let mut whisper = whisper_processor.lock().await;
  let mut audio_buffer = AudioBuffer::new(10000, 100, 500, sample_rate);

  let mut transcripts = Vec::new();
  let mut frame_buffer = Vec::<f32>::new();

  let vad_start = Instant::now();
  let mut whisper_total_time = Duration::ZERO;

  // Process in chunks
  for chunk in audio_data.chunks(1024) {
    frame_buffer.extend_from_slice(chunk);

    // Process 512-sample frames
    while frame_buffer.len() >= 512 {
      let frame: Vec<f32> = frame_buffer.drain(..512).collect();
      let speech_prob = vad.process_chunk(&frame)?;
      let is_speech = vad.is_speech(speech_prob);

      if let Some(complete_audio) = audio_buffer.add_chunk(&frame, is_speech) {
        // Measure Whisper transcription time
        let whisper_start = Instant::now();
        let transcript = whisper.transcribe(&complete_audio)?;
        whisper_total_time += whisper_start.elapsed();

        if !transcript.trim().is_empty() && !transcript.contains("[BLANK_AUDIO]") {
          transcripts.push(transcript.trim().to_string());
        }
      }
    }
  }

  stats.vad_processing_duration = vad_start.elapsed() - whisper_total_time;
  stats.whisper_transcription_duration = whisper_total_time;

  Ok(transcripts.join(" "))
}

// Create streaming transcription response
async fn create_transcription_stream(
  state: Arc<AppState>,
  model_name: String, // Change to owned String
  audio_data: Vec<f32>,
  mut stats: ProcessingStats,
) -> Result<impl Stream<Item = Result<Event, anyhow::Error>>, (StatusCode, Json<ErrorResponse>)> {
  let stream_start = Instant::now();

  // Get the appropriate Whisper processor for this model with timing
  let model_loading_start = Instant::now();
  let whisper_processor = match state.get_whisper_processor(&model_name).await {
    Ok(processor) => processor,
    Err(e) => {
      return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: ErrorDetail {
            message:    format!("Failed to load model '{}': {}", model_name, e),
            error_type: "invalid_request_error".to_string(),
            param:      Some("model".to_string()),
            code:       None,
          },
        }),
      ));
    },
  };
  stats.model_loading_duration = model_loading_start.elapsed();

  let sample_rate = 16000;

  Ok(stream::unfold((state, whisper_processor, audio_data, 0, AudioBuffer::new(10000, 100, 500, sample_rate), stats, stream_start), move |(state, whisper_processor, audio_data, mut processed, mut audio_buffer, mut stats, stream_start)| async move {
    if processed >= audio_data.len() {
      // Print final statistics for streaming
      stats.total_duration = stream_start.elapsed();
      stats.print_summary();
      return None;
    }

    // Process audio in chunks suitable for VAD (512 samples at a time)
    let chunk_size = 512.min(audio_data.len() - processed);
    let chunk = &audio_data[processed..processed + chunk_size];
    processed += chunk_size;

    // Process through VAD and Whisper processors
    let mut whisper_result = None;

    // Process through VAD
    let vad_chunk_start = Instant::now();
    let mut vad = state.vad.lock().await;
    if let Ok(speech_prob) = vad.process_chunk(chunk) {
      let is_speech = vad.is_speech(speech_prob);

      // Add to audio buffer and check if we have complete audio
      if let Some(complete_audio) = audio_buffer.add_chunk(chunk, is_speech) {
        // Release VAD lock before acquiring Whisper lock
        drop(vad);
        let vad_chunk_time = vad_chunk_start.elapsed();
        stats.vad_processing_duration += vad_chunk_time;

        // Process complete audio through Whisper
        let whisper_chunk_start = Instant::now();
        let mut whisper = whisper_processor.lock().await;
        if let Ok(transcript) = whisper.transcribe(&complete_audio) {
          let whisper_chunk_time = whisper_chunk_start.elapsed();
          stats.whisper_transcription_duration += whisper_chunk_time;

          if !transcript.trim().is_empty() && !transcript.contains("[BLANK_AUDIO]") {
            whisper_result = Some(transcript.trim().to_string());
            println!("🎯 Chunk transcribed in {:.2}ms: \"{}\"", whisper_chunk_time.as_secs_f64() * 1000.0, transcript.trim());
          }
        }
      }
    } else {
      stats.vad_processing_duration += vad_chunk_start.elapsed();
    }

    // Create event with actual transcription or progress update
    let event_data = if let Some(transcript) = whisper_result {
      StreamChunk { text: transcript, timestamp: Some(processed as f64 / sample_rate as f64) }
    } else {
      StreamChunk {
        text:      format!("Processing... ({:.1}%)", (processed as f64 / audio_data.len() as f64) * 100.0),
        timestamp: Some(processed as f64 / sample_rate as f64),
      }
    };

    let event = Event::default().json_data(event_data).unwrap();

    Some((Ok(event), (state.clone(), whisper_processor.clone(), audio_data, processed, audio_buffer, stats, stream_start)))
  }))
}
