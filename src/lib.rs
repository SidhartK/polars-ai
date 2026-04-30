use std::env;
use std::fs;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::future::join_all;
use futures::future::BoxFuture;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::runtime::Runtime;

// ---------------------------------------------------------------------------
// Kwargs passed from Python via register_plugin_function
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct MapKwargs {
    /// JSON string produced by AiModel.model_config
    model_config: String,
}

// ---------------------------------------------------------------------------
// Typed row context + provider interface
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AiContext {
    typ: String,
    value: String,
    mime: Option<String>,
    #[allow(dead_code)]
    meta: String,
}

#[derive(Debug)]
struct ModelError(String);

impl From<String> for ModelError {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ModelError {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

trait ModelProvider: Send + Sync {
    fn call(&self, input: AiContext) -> BoxFuture<'static, Result<String, ModelError>>;
}

// ---------------------------------------------------------------------------
// Fake provider — deterministic, zero network calls
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FakeProvider {
    prompt: String,
    tag: String,
}

impl ModelProvider for FakeProvider {
    fn call(&self, input: AiContext) -> BoxFuture<'static, Result<String, ModelError>> {
        let prompt = self.prompt.clone();
        let tag = self.tag.clone();
        Box::pin(async move {
            // Simulate minimal async latency so Tokio actually has something to
            // schedule concurrently.
            tokio::time::sleep(Duration::from_millis(1)).await;

            let rendered_prompt = prompt.replace("{value}", &input.value);

            Ok(format!(
                "[FAKE | tag={} | type={} | input_len={}] prompt=\"{}\" -> output=\"{}...\"",
                tag,
                input.typ,
                input.value.len(),
                &rendered_prompt[..rendered_prompt.len().min(40)],
                &input.value[..input.value.len().min(30)],
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// OpenAI provider — supports text, image URLs, local image paths, and b64
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct OpenAiOptions {
    max_output_tokens: Option<u64>,
}

#[derive(Clone, Debug)]
struct OpenAiProvider {
    prompt: String,
    model: String,
    options: OpenAiOptions,
    client: reqwest::Client,
}

fn render_openai_content(ctx: &AiContext, prompt: &str) -> Result<Value, ModelError> {
    let rendered_prompt = prompt.replace("{value}", &ctx.value);
    let mime = ctx.mime.as_deref().unwrap_or("image/jpeg");

    match ctx.typ.as_str() {
        "text" => Ok(json!(rendered_prompt)),
        "image_url" => Ok(json!([
            {"type": "text", "text": rendered_prompt},
            {"type": "image_url", "image_url": {"url": ctx.value}}
        ])),
        "image_path" => {
            let bytes = fs::read(&ctx.value).map_err(|e| {
                ModelError(format!("failed to read image path `{}`: {}", ctx.value, e))
            })?;
            let encoded = BASE64_STANDARD.encode(bytes);
            Ok(json!([
                {"type": "text", "text": rendered_prompt},
                {"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", mime, encoded)}}
            ]))
        }
        "image" => Ok(json!([
            {"type": "text", "text": rendered_prompt},
            {"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", mime, ctx.value)}}
        ])),
        other => Err(ModelError(format!(
            "unsupported AiModelContext _type `{}`",
            other
        ))),
    }
}

impl ModelProvider for OpenAiProvider {
    fn call(&self, input: AiContext) -> BoxFuture<'static, Result<String, ModelError>> {
        let prompt = self.prompt.clone();
        let model = self.model.clone();
        let options = self.options.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let api_key = match env::var("OPENAI_API_KEY") {
                Ok(key) if !key.trim().is_empty() => key,
                _ => return Err(ModelError("OPENAI_API_KEY is not set".to_string())),
            };

            let content = render_openai_content(&input, &prompt)?;

            // We intentionally keep the shared interface free of provider knobs.
            // OpenAI-specific options live inside OpenAiProvider.
            let mut body = json!({
                "model": model,
                "messages": [{"role": "user", "content": content}],
            });

            if let Some(max_out) = options.max_output_tokens {
                body["max_tokens"] = json!(max_out);
            }

            let response = client
                .post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| ModelError(format!("request failed: {}", e)))?;

            let status = response.status();
            let body_text = response
                .text()
                .await
                .map_err(|e| ModelError(format!("failed to read response body: {}", e)))?;

            if !status.is_success() {
                return Err(ModelError(format!("HTTP {}: {}", status, body_text)));
            }

            let parsed: Value = serde_json::from_str(&body_text)
                .map_err(|e| ModelError(format!("failed to parse response JSON: {}", e)))?;

            parsed
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    ModelError("response did not include choices[0].message.content".to_string())
                })
        })
    }
}

fn provider_from_config(cfg: &Value) -> Arc<dyn ModelProvider> {
    let provider = cfg
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("fake");

    match provider {
        "openai" => {
            let prompt = cfg
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("{value}")
                .to_string();
            let model = cfg
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("gpt-4o-mini")
                .to_string();

            // Forward-looking: prefer cfg.options.max_output_tokens
            // Back-compat: accept cfg.max_tokens
            let max_output_tokens = cfg
                .get("options")
                .and_then(|o| o.get("max_output_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| cfg.get("max_tokens").and_then(|v| v.as_u64()));

            Arc::new(OpenAiProvider {
                prompt,
                model,
                options: OpenAiOptions { max_output_tokens },
                client: reqwest::Client::new(),
            })
        }
        _ => {
            let prompt = cfg
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("{value}")
                .to_string();

            // Forward-looking: prefer cfg.options.tag
            // Back-compat: accept cfg.tag
            let tag = cfg
                .get("options")
                .and_then(|o| o.get("tag"))
                .and_then(|v| v.as_str())
                .or_else(|| cfg.get("tag").and_then(|v| v.as_str()))
                .unwrap_or("fake")
                .to_string();

            Arc::new(FakeProvider { prompt, tag })
        }
    }
}

// ---------------------------------------------------------------------------
// The Polars expression plugin — output is always Utf8 / String
// ---------------------------------------------------------------------------

#[polars_expr(output_type=String)]
fn ai_map(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];

    // Validate that we received a Struct with the four required fields
    let struct_series = series.struct_().map_err(|_| {
        PolarsError::InvalidOperation(
            "`.ctx.map()` requires an AiModelContext column (Struct dtype). \
             Use `pl_ai.text()` or `pl_ai.image_url()` to create one first."
                .into(),
        )
    })?;

    let fields = struct_series.fields_as_series();
    let field_names: Vec<&str> = fields.iter().map(|s: &Series| s.name().as_str()).collect();

    for required in &["_type", "_value", "_mime", "_meta"] {
        if !field_names.contains(required) {
            return Err(PolarsError::InvalidOperation(
                format!(
                    "`.ctx.map()` requires AiModelContext field `{}` but it was not found. \
                     Fields present: {:?}",
                    required, field_names
                )
                .into(),
            ));
        }
    }

    // Build a rate limiter: 50 requests per second
    let quota = Quota::per_second(NonZeroU32::new(50).unwrap());
    let limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>> =
        Arc::new(RateLimiter::direct(quota));

    // Extract the four fields
    let type_series = struct_series.field_by_name("_type")?;
    let value_series = struct_series.field_by_name("_value")?;
    let mime_series = struct_series.field_by_name("_mime")?;
    let meta_series = struct_series.field_by_name("_meta")?;

    let row_count = series.len();
    let mut row_contexts: Vec<AiContext> = Vec::with_capacity(row_count);

    for i in 0..row_count {
        let t = type_series.str()?.get(i).unwrap_or("null");
        let v = value_series.str()?.get(i).unwrap_or("null");
        let m = mime_series.str()?.get(i);
        let e = meta_series.str()?.get(i).unwrap_or("{}");
        let mime = m.and_then(|s| {
            if s == "null" {
                None
            } else {
                Some(s.to_string())
            }
        });

        row_contexts.push(AiContext {
            typ: t.to_string(),
            value: v.to_string(),
            mime,
            meta: e.to_string(),
        });
    }

    // Spin up a multi-threaded Tokio runtime and fan out all rows concurrently
    let rt = Runtime::new().map_err(|e| {
        PolarsError::ComputeError(format!("failed to create tokio runtime: {}", e).into())
    })?;

    let model_config = kwargs.model_config.clone();
    let cfg: serde_json::Value = serde_json::from_str(&model_config).unwrap_or(Value::Null);
    let provider: Arc<dyn ModelProvider> = provider_from_config(&cfg);

    let results: Vec<String> = rt.block_on(async {
        let futures: Vec<_> = row_contexts
            .into_iter()
            .map(|ctx| {
                let limiter = limiter.clone();
                let provider = provider.clone();
                async move {
                    // Wait for a rate-limit token before proceeding
                    limiter.until_ready().await;
                    match provider.call(ctx).await {
                        Ok(s) => s,
                        Err(e) => format!("[MODEL ERROR] {}", e.0),
                    }
                }
            })
            .collect();

        join_all(futures).await
    });

    // Assemble the output Series
    let ca = StringChunked::from_iter_options(
        PlSmallStr::from_static("ai_output"),
        results.iter().map(|s| Some(s.as_str())),
    );

    Ok(ca.into_series())
}

// ---------------------------------------------------------------------------
// pyo3 module entry point — required by maturin / pyo3-polars
// ---------------------------------------------------------------------------

#[pymodule]
fn _polars_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
