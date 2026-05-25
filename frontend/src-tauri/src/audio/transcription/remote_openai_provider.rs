// audio/transcription/remote_openai_provider.rs
//
// Transcription provider that posts audio to any HTTP endpoint implementing
// OpenAI's /audio/transcriptions contract (self-hosted whisper.cpp servers,
// Groq, OpenAI, an Ollama box fronting whisper, etc.).

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::info;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::time::Duration;

const SAMPLE_RATE: u32 = 16_000;

pub struct RemoteOpenAiProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

impl RemoteOpenAiProvider {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        let trimmed = base_url.trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            base_url: trimmed,
            model,
            api_key,
            client,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url)
    }
}

#[async_trait]
impl TranscriptionProvider for RemoteOpenAiProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if audio.is_empty() {
            return Err(TranscriptionError::AudioTooShort {
                samples: 0,
                minimum: 1,
            });
        }

        let wav_bytes = encode_wav_16k_mono(&audio);

        let file_part = Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| TranscriptionError::EngineFailed(format!("multipart mime: {}", e)))?;

        let mut form = Form::new()
            .text("model", self.model.clone())
            .text("response_format", "json")
            .part("file", file_part);
        if let Some(lang) = language {
            form = form.text("language", lang);
        }

        let mut req = self.client.post(self.endpoint()).multipart(form);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }

        let response = req
            .send()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(TranscriptionError::EngineFailed(format!(
                "{}: {}",
                status,
                truncate_for_log(&body)
            )));
        }

        let parsed: TranscriptionResponse = response.json().await.map_err(|e| {
            TranscriptionError::EngineFailed(format!("response parse failed: {}", e))
        })?;

        Ok(TranscriptResult {
            text: parsed.text.trim().to_string(),
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        !self.base_url.is_empty() && !self.model.is_empty()
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn provider_name(&self) -> &'static str {
        "Remote OpenAI-Compatible"
    }
}

/// Encode 16 kHz mono f32 PCM samples into a WAV byte buffer (16-bit signed PCM).
fn encode_wav_16k_mono(samples: &[f32]) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let bits_per_sample: u16 = 16;
    let num_channels: u16 = 1;
    let byte_rate = SAMPLE_RATE * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples * (bits_per_sample as u32 / 8);
    let chunk_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.max(-1.0).min(1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 300;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let mut t = s[..MAX].to_string();
        t.push_str("…");
        t
    }
}

#[allow(dead_code)]
pub async fn ping_remote_openai(base_url: &str) -> Result<(), String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    info!("Pinging remote OpenAI-compatible endpoint: {}", url);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("client init: {}", e))?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("connection failed: {}", e))?;
    if resp.status().is_success() || resp.status().as_u16() == 401 {
        // 401 is acceptable — endpoint reachable but rejects unauthenticated GET.
        Ok(())
    } else {
        Err(format!("endpoint returned {}", resp.status()))
    }
}
