use std::{collections::HashMap, sync::Arc};

use crate::AppState;
use crate::api::{default_model, default_response_format, default_temperature, ErrorDetail, ErrorResponse, StreamChunk, TranscriptionRequest, TranscriptionResponse};

use crate::audio_manager::AudioBuffer;
use anyhow::Result;
use axum::{
  Json,
  extract::{Multipart, Query, State},
  http::StatusCode,
  response::{
    IntoResponse, Response,
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

// Main transcription endpoint
pub async fn transcribe_audio(
  State(state): State<Arc<AppState>>,
  Query(params): Query<HashMap<String, String>>,
  mut multipart: Multipart,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
  // Parse query parameters for streaming
  let stream_enabled = params
    .get("stream")
    .map(|s| s.parse::<bool>().unwrap_or(false))
    .unwrap_or(false);

  // Extract audio file from multipart form
  let audio_data = match extract_audio_from_multipart(&mut multipart).await {
    Ok(data) => data,
    Err(e) => {
      return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: ErrorDetail {
            message: format!("Failed to extract audio file: {}", e),
            error_type: "invalid_request_error".to_string(),
            param: Some("file".to_string()),
            code: None,
          },
        }),
      ));
    },
  };

  // Parse request parameters
  let request = TranscriptionRequest {
    model: params
      .get("model")
      .cloned()
      .unwrap_or_else(default_model),
    language: params.get("language").cloned(),
    prompt: params.get("prompt").cloned(),
    response_format: params
      .get("response_format")
      .cloned()
      .unwrap_or_else(default_response_format),
    temperature: params
      .get("temperature")
      .and_then(|s| s.parse().ok())
      .unwrap_or_else(default_temperature),
    stream: stream_enabled,
  };

  println!("Request: {:?}", request);

  // Convert audio to PCM format
  let pcm_data = match convert_audio_to_pcm(&audio_data).await {
    Ok(data) => data,
    Err(e) => {
      return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: ErrorDetail {
            message: format!("Failed to process audio file: {}", e),
            error_type: "invalid_request_error".to_string(),
            param: Some("file".to_string()),
            code: None,
          },
        }),
      ));
    },
  };

  println!("Audio data length: {:?}", pcm_data.len());

  if request.stream {
    // Return streaming response
    let stream = create_transcription_stream(state, pcm_data).await;
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());

    Ok(sse.into_response())
  } else {
    // Return single response
    match transcribe_audio_complete(state, pcm_data).await {
      Ok(transcript) => Ok(Json(TranscriptionResponse { text: transcript }).into_response()),
      Err(e) => Err((
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: ErrorDetail {
            message: format!("Transcription failed: {}", e),
            error_type: "server_error".to_string(),
            param: None,
            code: None,
          },
        }),
      )),
    }
  }
}

// Extract audio file from multipart form data
async fn extract_audio_from_multipart(multipart: &mut Multipart) -> Result<Vec<u8>> {
  while let Some(field) = multipart.next_field().await? {
    if let Some(name) = field.name() {
      if name == "file" {
        let data = field.bytes().await?;
        return Ok(data.to_vec());
      }
    }
  }
  anyhow::bail!("No file field found in multipart data")
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
pub async fn transcribe_audio_complete(
  state: Arc<AppState>,
  audio_data: Vec<f32>,
) -> Result<String> {
  let sample_rate = 16000;

  // Process audio through VAD and Whisper
  let mut vad = state.vad.lock().await;
  let mut whisper = state.whisper.lock().await;
  let mut audio_buffer = AudioBuffer::new(10000, 100, 500, sample_rate);

  let mut transcripts = Vec::new();
  let mut frame_buffer = Vec::<f32>::new();

  // Process in chunks
  for chunk in audio_data.chunks(1024) {
    frame_buffer.extend_from_slice(chunk);

    // Process 512-sample frames
    while frame_buffer.len() >= 512 {
      let frame: Vec<f32> = frame_buffer.drain(..512).collect();
      let speech_prob = vad.process_chunk(&frame)?;
      let is_speech = vad.is_speech(speech_prob);

      if let Some(complete_audio) = audio_buffer.add_chunk(&frame, is_speech) {
        let transcript = whisper.transcribe(&complete_audio)?;
        if !transcript.trim().is_empty() && !transcript.contains("[BLANK_AUDIO]") {
          transcripts.push(transcript.trim().to_string());
        }
      }
    }
  }

  Ok(transcripts.join(" "))
}

// Create streaming transcription response
pub async fn create_transcription_stream(
  state: Arc<AppState>,
  audio_data: Vec<f32>,
) -> impl Stream<Item = Result<Event, anyhow::Error>> {
  let sample_rate = 16000;

  stream::unfold((state, audio_data, 0, AudioBuffer::new(10000, 100, 500, sample_rate)), move |(state, audio_data, mut processed, mut audio_buffer)| async move {
    if processed >= audio_data.len() {
      return None;
    }

    // Process audio in chunks suitable for VAD (512 samples at a time)
    let chunk_size = 512.min(audio_data.len() - processed);
    let chunk = &audio_data[processed..processed + chunk_size];
    processed += chunk_size;

    // Process through VAD and Whisper processors
    let mut whisper_result = None;

    // Process through VAD
    let mut vad = state.vad.lock().await;
    if let Ok(speech_prob) = vad.process_chunk(chunk) {
      let is_speech = vad.is_speech(speech_prob);

      // Add to audio buffer and check if we have complete audio
      if let Some(complete_audio) = audio_buffer.add_chunk(chunk, is_speech) {
        // Release VAD lock before acquiring Whisper lock
        drop(vad);

        // Process complete audio through Whisper
        let mut whisper = state.whisper.lock().await;
        if let Ok(transcript) = whisper.transcribe(&complete_audio) {
          if !transcript.trim().is_empty() && !transcript.contains("[BLANK_AUDIO]") {
            whisper_result = Some(transcript.trim().to_string());
          }
        }
      }
    }

    // Create event with actual transcription or progress update
    let event_data = if let Some(transcript) = whisper_result {
      StreamChunk { text: transcript, timestamp: Some(processed as f64 / sample_rate as f64) }
    } else {
      StreamChunk {
        text: format!("Processing... ({:.1}%)", (processed as f64 / audio_data.len() as f64) * 100.0),
        timestamp: Some(processed as f64 / sample_rate as f64),
      }
    };

    let event = Event::default().json_data(event_data).unwrap();

    Some((Ok(event), (state.clone(), audio_data, processed, audio_buffer)))
  })
}
