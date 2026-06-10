//! Cloud-based ASR backends (currently 0G Compute Router Whisper-large-v3).
//!
//! Provides a thin OpenAI-compatible Whisper client suitable for transcribing
//! short utterances captured by the local audio pipeline. Unlike the local
//! Whisper.cpp path, audio is shipped over HTTPS to a 0G TEE provider where
//! it is decrypted and processed inside an Intel TDX + NVIDIA H100/H200
//! enclave; the provider operator physically cannot inspect raw audio.
//!
//! Design notes:
//!
//! * The request shape mirrors the OpenAI Whisper API
//!   (`POST /audio/transcriptions` with `multipart/form-data`) so we can reuse
//!   battle-tested server-side code paths on the 0G router side.
//! * Bearer auth uses the 0G Compute API key from the dashboard
//!   (https://pc.0g.ai). Settlement happens off-band via the per-account 0G
//!   token balance, so we do NOT need to sign on-chain transactions per call.
//! * The TEE proof field name (`tee_proof`) is best-guess based on the public
//!   0G Private Computer blog post and will be reconciled against the live
//!   response on the first real call. Missing/renamed fields degrade to
//!   `proof = None` rather than failing the whole request, unless
//!   `require_tee_proof` is set.
//!
//! Public entry point: [`transcribe_0g_whisper`].

use anyhow::{anyhow, Context, Result};
use log::debug;
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Default base URL for the 0G Compute Router (OpenAI-compatible).
pub const DEFAULT_0G_BASE_URL: &str = "https://router-api.0g.ai/v1";

/// Default ASR model identifier on 0G.
pub const DEFAULT_0G_ASR_MODEL: &str = "whisper-large-v3";

/// Recommended Whisper `initial_prompt` for Mandarin Chinese.
///
/// Whisper systematically omits Chinese punctuation when transcribing
/// (see https://github.com/ggml-org/whisper.cpp/issues/2532). Supplying an
/// initial-prompt that already contains punctuation biases the decoder
/// toward emitting punctuation in the output, which materially improves
/// the raw transcript. This is a partial fix — full LLM cleanup is still
/// recommended for production.
pub const ZH_INITIAL_PROMPT: &str = "以下是一段普通话语音，请输出带标点的转写。";

/// Result returned by a successful cloud ASR call.
#[derive(Debug, Clone)]
pub struct CloudAsrResult {
    /// The transcribed text. For Whisper-large-v3, the model may omit
    /// punctuation entirely when transcribing Chinese — callers are
    /// expected to feed `text` through a post-processing LLM if punctuation
    /// is desired.
    pub text: String,

    /// Detected language (BCP-47 style code, e.g. "zh", "en") if the
    /// provider returned one. May be `None` when `language` was explicitly
    /// forced.
    pub language: Option<String>,

    /// Optional TEE attestation proof attached by the 0G provider. When
    /// `require_tee_proof` is true and this field is `None`, the call is
    /// rejected.
    pub proof: Option<TeeProof>,

    /// End-to-end wall-clock latency in milliseconds.
    pub latency_ms: u128,
}

/// TEE attestation metadata returned by 0G providers running in `TeeML` mode.
///
/// Field shapes are inferred from the public 0G Private Computer
/// documentation (https://0g.ai/blog/0g-private-computer) and will need to
/// be reconciled against the actual response once the first real provider
/// call is made. All fields default to empty / `None` so an unexpected shape
/// degrades gracefully.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TeeProof {
    /// Address of the on-chain signer for the enclave.
    #[serde(default)]
    pub signer_address: String,
    /// Hash of the docker-compose manifest of the enclave runtime.
    #[serde(default)]
    pub compose_hash: String,
    /// Signature over the response payload produced inside the enclave.
    #[serde(default)]
    pub signature: String,
    /// Hash of the model weights served by the enclave.
    #[serde(default)]
    pub model_hash: Option<String>,
}

/// Configuration passed to [`transcribe_0g_whisper`].
///
/// All fields except `audio_wav` are typically derived from the persisted
/// `PostProcessProvider` entry whose `id == "zerog"` and the corresponding
/// `post_process_api_keys` secret. Use the builder methods to construct.
#[derive(Debug, Clone)]
pub struct CloudAsrRequest {
    /// Base URL of the OpenAI-compatible router
    /// (e.g. `https://router-api.0g.ai/v1`).
    pub base_url: String,
    /// Bearer token obtained from the 0G Compute dashboard.
    pub api_key: String,
    /// Model identifier (e.g. `whisper-large-v3`).
    pub model: String,
    /// WAV-encoded mono PCM audio bytes. Sample rate must match what the
    /// provider expects (Whisper accepts arbitrary, but 16 kHz / 16-bit is
    /// idiomatic and what Handy already produces).
    pub audio_wav: Vec<u8>,
    /// BCP-47 language hint. `None` or `Some("auto")` enables language
    /// detection. Forcing `"zh"` or `"en"` significantly improves accuracy
    /// for short utterances.
    pub language: Option<String>,
    /// Whisper `initial_prompt`. For Chinese, supplying
    /// [`ZH_INITIAL_PROMPT`] partially mitigates the well-known issue that
    /// Whisper omits Chinese punctuation by default.
    pub initial_prompt: Option<String>,
    /// When true, the call fails if the provider response lacks a
    /// `tee_proof` field. Use this for "TEE-only" mode where downgrading
    /// to plain HTTPS inference is unacceptable.
    pub require_tee_proof: bool,
    /// Per-request timeout. Defaults to 60s; raise for long utterances.
    pub timeout: Duration,
}

impl CloudAsrRequest {
    /// Construct a new request with sensible defaults
    /// (`whisper-large-v3`, no language forcing, 60s timeout, TEE optional).
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        audio_wav: Vec<u8>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: DEFAULT_0G_ASR_MODEL.to_string(),
            audio_wav,
            language: None,
            initial_prompt: None,
            require_tee_proof: false,
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    pub fn with_initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = Some(prompt.into());
        self
    }

    pub fn with_require_tee_proof(mut self, require: bool) -> Self {
        self.require_tee_proof = require;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    /// Optional TEE proof. Field name guessed from 0G docs; verified at
    /// runtime to be either present or `None` rather than failing on
    /// schema mismatch.
    #[serde(default, rename = "tee_proof")]
    tee_proof: Option<TeeProof>,
}

/// Transcribes a single WAV utterance via 0G Compute Router
/// (Whisper-large-v3).
///
/// Sends a multipart `POST /audio/transcriptions` request matching the
/// OpenAI Whisper API surface. The 0G router accepts the same shape and
/// forwards the payload to a TEE provider; the response includes optional
/// TEE attestation metadata when the underlying provider runs in `TeeML`
/// mode.
///
/// Errors are returned as `anyhow::Error` carrying both the HTTP status
/// (when applicable) and a truncated response body for debugging.
pub async fn transcribe_0g_whisper(req: CloudAsrRequest) -> Result<CloudAsrResult> {
    if req.api_key.trim().is_empty() {
        return Err(anyhow!("0G Compute API key is empty"));
    }
    if req.audio_wav.is_empty() {
        return Err(anyhow!("audio buffer is empty"));
    }

    let url = format!(
        "{}/audio/transcriptions",
        req.base_url.trim_end_matches('/')
    );
    debug!(
        "0G ASR request: url={} model={} bytes={} lang={:?}",
        url,
        req.model,
        req.audio_wav.len(),
        req.language
    );

    let mut form = multipart::Form::new()
        .text("model", req.model.clone())
        .text("response_format", "json");

    // Force language when supplied; "auto" / None lets Whisper detect.
    if let Some(lang) = req
        .language
        .as_deref()
        .filter(|l| !l.is_empty() && *l != "auto")
    {
        form = form.text("language", lang.to_string());
    }

    if let Some(prompt) = req.initial_prompt.as_deref().filter(|p| !p.is_empty()) {
        form = form.text("prompt", prompt.to_string());
    }

    let part = multipart::Part::bytes(req.audio_wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .context("setting multipart MIME type")?;
    form = form.part("file", part);

    let client = reqwest::Client::builder()
        .timeout(req.timeout)
        .build()
        .context("building reqwest client for 0G ASR")?;

    let start = Instant::now();
    let resp = client
        .post(&url)
        .bearer_auth(&req.api_key)
        .multipart(form)
        .send()
        .await
        .context("sending 0G ASR request")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "0G ASR HTTP {}: {}",
            status,
            body.chars().take(512).collect::<String>()
        ));
    }

    let parsed: ApiResponse = resp.json().await.context("decoding 0G ASR JSON response")?;

    if req.require_tee_proof && parsed.tee_proof.is_none() {
        return Err(anyhow!(
            "TEE attestation required but provider response had no tee_proof"
        ));
    }

    let latency = start.elapsed().as_millis();
    debug!(
        "0G ASR succeeded in {} ms; text_len={} lang={:?} proof_present={}",
        latency,
        parsed.text.len(),
        parsed.language,
        parsed.tee_proof.is_some()
    );

    Ok(CloudAsrResult {
        text: parsed.text,
        language: parsed.language,
        proof: parsed.tee_proof,
        latency_ms: latency,
    })
}

/// High-level convenience wrapper used by [`crate::actions`] when the user
/// has flipped `cloud_asr_enabled` in settings.
///
/// Pulls everything cloud ASR needs out of [`crate::settings::AppSettings`]
/// (provider URL, API key, target model, TEE strictness, language hint,
/// custom vocabulary) and returns just the transcribed text, mirroring the
/// signature of the local [`crate::managers::transcription::TranscriptionManager::transcribe`]
/// so the call site stays a one-liner.
///
/// Lives here rather than in `actions.rs` so the call site does not need to
/// learn about WAV encoding, language code normalisation, or initial-prompt
/// construction — those are cloud-ASR concerns.
pub async fn transcribe_with_app_settings(
    audio_samples: Vec<f32>,
    settings: &crate::settings::AppSettings,
) -> Result<String> {
    use crate::audio_toolkit::encode_wav_bytes;

    let provider_id = settings.cloud_asr_provider_id.clone();

    let provider = settings
        .post_process_provider(&provider_id)
        .ok_or_else(|| {
            anyhow!(
                "cloud ASR provider '{}' is not registered in post_process_providers",
                provider_id
            )
        })?;

    let api_key = settings
        .post_process_api_keys
        .get(&provider_id)
        .filter(|k| !k.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "cloud ASR provider '{}' has no API key configured",
                provider_id
            )
        })?
        .clone();

    let wav =
        encode_wav_bytes(&audio_samples).context("encoding samples to WAV for cloud ASR upload")?;

    // Map Handy's language codes to the BCP-47 subset Whisper understands.
    // Mirrors the local-engine logic in `managers/transcription.rs` so cloud
    // and local pipelines never disagree on what language the user picked.
    let language: Option<String> = match settings.selected_language.as_str() {
        "auto" | "" => None,
        "zh-Hans" | "zh-Hant" | "zh-CN" | "zh-TW" => Some("zh".to_string()),
        other => Some(other.to_string()),
    };

    let mut req = CloudAsrRequest::new(&provider.base_url, &api_key, wav)
        .with_model(&settings.cloud_asr_model)
        .with_require_tee_proof(settings.cloud_asr_require_tee_proof);

    if let Some(ref lang) = language {
        req = req.with_language(lang.clone());
        // Whisper omits Chinese punctuation by default; biasing with an
        // initial prompt that already contains punctuation materially
        // improves the raw transcript. This is a partial fix — LLM cleanup
        // remains the production-grade answer.
        if lang == "zh" {
            req = req.with_initial_prompt(ZH_INITIAL_PROMPT);
        }
    }

    // Fold any user-defined custom vocabulary into the initial prompt so it
    // affects cloud ASR the same way it affects the local Whisper engine
    // (see `WhisperInferenceParams.initial_prompt` in transcription.rs).
    if !settings.custom_words.is_empty() {
        let extras = settings.custom_words.join(", ");
        let combined = match req.initial_prompt.as_deref() {
            Some(existing) => format!("{} 自定义词汇：{}.", existing, extras),
            None => format!("Use these custom words when relevant: {}.", extras),
        };
        req = req.with_initial_prompt(combined);
    }

    let result = transcribe_0g_whisper(req).await?;
    Ok(result.text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_defaults() {
        let req = CloudAsrRequest::new("https://x/v1", "k", vec![0u8; 8]);
        assert_eq!(req.model, DEFAULT_0G_ASR_MODEL);
        assert!(req.language.is_none());
        assert!(req.initial_prompt.is_none());
        assert!(!req.require_tee_proof);
        assert_eq!(req.timeout, Duration::from_secs(60));
    }

    #[test]
    fn builder_chains() {
        let req = CloudAsrRequest::new("https://x/v1", "k", vec![0u8; 8])
            .with_model("whisper-large-v3-turbo")
            .with_language("zh")
            .with_initial_prompt(ZH_INITIAL_PROMPT)
            .with_require_tee_proof(true)
            .with_timeout(Duration::from_secs(30));
        assert_eq!(req.model, "whisper-large-v3-turbo");
        assert_eq!(req.language.as_deref(), Some("zh"));
        assert_eq!(req.initial_prompt.as_deref(), Some(ZH_INITIAL_PROMPT));
        assert!(req.require_tee_proof);
        assert_eq!(req.timeout, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn rejects_empty_api_key() {
        let req = CloudAsrRequest::new("https://x/v1", "", vec![0u8; 8]);
        let err = transcribe_0g_whisper(req).await.unwrap_err();
        assert!(err.to_string().contains("API key"));
    }

    #[tokio::test]
    async fn rejects_empty_audio() {
        let req = CloudAsrRequest::new("https://x/v1", "k", vec![]);
        let err = transcribe_0g_whisper(req).await.unwrap_err();
        assert!(err.to_string().contains("audio"));
    }
}
