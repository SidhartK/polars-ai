//! polars-ai: expression plugins for model context mapping and hydration.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use chrono::{SecondsFormat, Utc};
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use governor::{Quota, RateLimiter};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

// -----------------------------------------------------------------------------
// Mirror polars_ai.types (RESPONSE_SCHEMA_VERSION)
// -----------------------------------------------------------------------------
const CACHE_SCHEMA_VERSION: &str = "2";
const STAT_OK: &str = "ok";
const STAT_CACHE_HIT: &str = "cache_hit";
const STAT_BUDGET_EXHAUSTED: &str = "budget_exhausted";
const STAT_MODEL_ERROR: &str = "model_error";
const STAT_INVALID_CTX: &str = "invalid_context";

// -----------------------------------------------------------------------------
// Plugin kwargs (JSON from Python register_plugin_function)
// -----------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone)]
struct MapKwargs {
    #[serde(rename = "model_config")]
    model_config: String,
    #[serde(default)]
    max_requests: Option<usize>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    max_concurrency: Option<usize>,
    #[serde(default)]
    cache_enabled: Option<bool>,
    #[serde(default)]
    cache_path: Option<String>,
    #[serde(default)]
    rate_limit_per_second: Option<u32>,
    #[serde(default)]
    multimodal: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
struct ReduceTextKwargs {
    #[serde(default = "default_text_separator")]
    text_separator: String,
    #[serde(default)]
    number_text_items: bool,
}

fn default_text_separator() -> String {
    "\n\n".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheDiskRow {
    cache_key: String,
    status: String,
    value: Option<String>,
    error: Option<String>,
    model_config: String,
    attempts: u32,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    cost_usd: Option<f64>,
    created_at: String,
    completed_at: String,
}

// -----------------------------------------------------------------------------
// Context + Provider
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AiContextAtom {
    typ: String,
    value: String,
    mime: Option<String>,
    #[allow(dead_code)]
    meta: String,
    value_is_null: bool,
}

type AiContext = AiContextAtom;

#[derive(Clone, Debug)]
struct AiContextBatch {
    items: Vec<AiContextAtom>,
}

#[derive(Clone, Debug)]
enum ModelInput {
    Atom(AiContextAtom),
    Batch(AiContextBatch),
}

#[derive(Debug)]
struct ModelError(String);

#[derive(Clone, Debug, Default)]
struct ModelOutput {
    text: String,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cost_usd: Option<f64>,
}

#[derive(Clone, Debug)]
struct TokenPricing {
    input_per_million: f64,
    output_per_million: f64,
}

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
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>>;
}

#[derive(Clone, Debug)]
struct FakeProvider {
    prompt: String,
    tag: String,
}

impl ModelProvider for FakeProvider {
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>> {
        let prompt = self.prompt.clone();
        let tag = self.tag.clone();
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            let atom = render_fake_input(input)?;
            let input_tokens = estimate_tokens(&atom);
            let rendered_prompt = prompt.replace("{value}", &atom.value);
            let text = format!(
                "[FAKE | tag={} | type={} | input_len={}] prompt=\"{}\" -> output=\"{}...\"",
                tag,
                atom.typ,
                atom.value.len(),
                &rendered_prompt[..rendered_prompt.len().min(40)],
                &atom.value[..atom.value.len().min(30)],
            );
            let output_tokens = estimate_text_tokens(&text);
            Ok(ModelOutput {
                text,
                input_tokens: Some(input_tokens),
                output_tokens: Some(output_tokens),
                total_tokens: Some(input_tokens.saturating_add(output_tokens)),
                cost_usd: Some(0.0),
            })
        })
    }
}

#[derive(Clone, Debug, Default)]
struct OpenAiOptions {
    max_output_tokens: Option<u64>,
}

#[derive(Clone, Debug)]
struct OpenAiProvider {
    prompt: String,
    model: String,
    options: OpenAiOptions,
    pricing: Option<TokenPricing>,
    client: reqwest::Client,
}

fn is_text_joinable(typ: &str) -> bool {
    matches!(typ, "text" | "json" | "blob" | "unknown")
}

fn reduce_batch_to_text_atom(
    batch: &AiContextBatch,
    text_separator: &str,
    number_text_items: bool,
) -> AiContextAtom {
    if batch.items.iter().any(|item| item.value_is_null) {
        return AiContextAtom {
            typ: "text".to_string(),
            value: String::new(),
            mime: None,
            meta: json!({"reduction": "text", "error": "null _value in ContextBatch"})
                .to_string(),
            value_is_null: true,
        };
    }

    if let Some(item) = batch
        .items
        .iter()
        .find(|item| !is_text_joinable(item.typ.as_str()))
    {
        return AiContextAtom {
            typ: "unsupported_batch".to_string(),
            value: format!(
                "ContextBatch contains `{}`; use native multimodal batch mapping instead of reduce_text",
                item.typ
            ),
            mime: None,
            meta: json!({"reduction": "text", "unsupported_type": item.typ}).to_string(),
            value_is_null: false,
        };
    }

    let parts: Vec<String> = batch
        .items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            if number_text_items {
                format!("Item {}:\n{}", idx + 1, item.value)
            } else {
                item.value.clone()
            }
        })
        .collect();

    AiContextAtom {
        typ: "text".to_string(),
        value: parts.join(text_separator),
        mime: None,
        meta: json!({
            "reduction": "text",
            "text_separator": text_separator,
            "number_text_items": number_text_items
        })
        .to_string(),
        value_is_null: false,
    }
}

fn render_fake_input(input: ModelInput) -> Result<AiContextAtom, ModelError> {
    match input {
        ModelInput::Atom(ctx) => {
            if ctx.typ == "unsupported_batch" {
                Err(ModelError(ctx.value))
            } else {
                Ok(ctx)
            }
        }
        ModelInput::Batch(batch) => {
            if let Some(item) = batch
                .items
                .iter()
                .find(|item| !is_text_joinable(item.typ.as_str()))
            {
                return Err(ModelError(format!(
                    "FakeModel does not support native multimodal ContextBatch item `{}`",
                    item.typ
                )));
            }
            let atom = reduce_batch_to_text_atom(&batch, "\n\n", false);
            if atom.typ == "unsupported_batch" {
                Err(ModelError(atom.value))
            } else {
                Ok(AiContextAtom {
                    typ: "batch".to_string(),
                    ..atom
                })
            }
        }
    }
}

fn render_openai_atom_content(ctx: &AiContextAtom, prompt: &str) -> Result<Value, ModelError> {
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
        "json" | "blob" | "unknown" => Ok(json!(rendered_prompt)),
        other => Err(ModelError(format!(
            "unsupported AiModelContext _type `{}`",
            other
        ))),
    }
}

fn render_openai_batch_content(batch: &AiContextBatch, prompt: &str) -> Result<Value, ModelError> {
    let mut parts: Vec<Value> = Vec::new();
    let text_value = batch
        .items
        .iter()
        .filter(|item| is_text_joinable(item.typ.as_str()))
        .map(|item| item.value.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let rendered_prompt = prompt.replace("{value}", &text_value);

    if !rendered_prompt.trim().is_empty() {
        parts.push(json!({"type": "text", "text": rendered_prompt}));
    }

    for item in &batch.items {
        let mime = item.mime.as_deref().unwrap_or("image/jpeg");
        match item.typ.as_str() {
            "text" | "json" | "blob" | "unknown" => {}
            "image_url" => {
                parts.push(json!({"type": "image_url", "image_url": {"url": item.value}}));
            }
            "image_path" => {
                let bytes = fs::read(&item.value).map_err(|e| {
                    ModelError(format!("failed to read image path `{}`: {}", item.value, e))
                })?;
                let encoded = BASE64_STANDARD.encode(bytes);
                parts.push(json!({"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", mime, encoded)}}));
            }
            "image" => {
                parts.push(json!({"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", mime, item.value)}}));
            }
            other => {
                return Err(ModelError(format!(
                    "unsupported AiModelContext _type `{}` in ContextBatch",
                    other
                )));
            }
        }
    }

    Ok(json!(parts))
}

fn render_openai_content(input: &ModelInput, prompt: &str) -> Result<Value, ModelError> {
    match input {
        ModelInput::Atom(ctx) => render_openai_atom_content(ctx, prompt),
        ModelInput::Batch(batch) => render_openai_batch_content(batch, prompt),
    }
}

impl ModelProvider for OpenAiProvider {
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>> {
        let prompt = self.prompt.clone();
        let model = self.model.clone();
        let options = self.options.clone();
        let pricing = self.pricing.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let api_key = match env::var("OPENAI_API_KEY") {
                Ok(key) if !key.trim().is_empty() => key,
                _ => return Err(ModelError("OPENAI_API_KEY is not set".to_string())),
            };

            let content = render_openai_content(&input, &prompt)?;

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

            let usage = parsed.get("usage");
            let input_tokens = usage
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| Some(estimate_model_input(&input)));
            let output_tokens = usage
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_u64());
            let total_tokens = usage
                .and_then(|u| u.get("total_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| match (input_tokens, output_tokens) {
                    (Some(i), Some(o)) => Some(i.saturating_add(o)),
                    _ => None,
                });

            let text = parsed
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    ModelError("response did not include choices[0].message.content".to_string())
                })?;

            let output_tokens = output_tokens.or_else(|| Some(estimate_text_tokens(&text)));
            let total_tokens = total_tokens.or_else(|| match (input_tokens, output_tokens) {
                (Some(i), Some(o)) => Some(i.saturating_add(o)),
                _ => None,
            });
            let cost_usd = calculate_cost_usd(input_tokens, output_tokens, pricing.as_ref());

            Ok(ModelOutput {
                text,
                input_tokens,
                output_tokens,
                total_tokens,
                cost_usd,
            })
        })
    }
}

fn number_from_config(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key)
        .and_then(|v| v.as_f64())
        .or_else(|| {
            cfg.get("options")
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
        })
        .or_else(|| {
            cfg.get("pricing")
                .and_then(|p| p.get(key))
                .and_then(|v| v.as_f64())
        })
}

fn pricing_from_config(cfg: &Value) -> Option<TokenPricing> {
    let input_per_million = number_from_config(cfg, "input_cost_per_1m_tokens")
        .or_else(|| number_from_config(cfg, "input_cost_per_million_tokens"));
    let output_per_million = number_from_config(cfg, "output_cost_per_1m_tokens")
        .or_else(|| number_from_config(cfg, "output_cost_per_million_tokens"));

    match (input_per_million, output_per_million) {
        (Some(input), Some(output)) => Some(TokenPricing {
            input_per_million: input,
            output_per_million: output,
        }),
        _ => cfg
            .get("model")
            .and_then(|v| v.as_str())
            .and_then(default_pricing_for_model),
    }
}

fn default_pricing_for_model(model: &str) -> Option<TokenPricing> {
    match model {
        "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => Some(TokenPricing {
            input_per_million: 0.15,
            output_per_million: 0.60,
        }),
        "gpt-4o" | "gpt-4o-2024-08-06" => Some(TokenPricing {
            input_per_million: 2.50,
            output_per_million: 10.00,
        }),
        _ => None,
    }
}

fn calculate_cost_usd(
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    pricing: Option<&TokenPricing>,
) -> Option<f64> {
    let pricing = pricing?;
    let input = input_tokens? as f64;
    let output = output_tokens? as f64;
    Some((input * pricing.input_per_million + output * pricing.output_per_million) / 1_000_000.0)
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
            let max_output_tokens = cfg
                .get("options")
                .and_then(|o| o.get("max_output_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| cfg.get("max_tokens").and_then(|v| v.as_u64()));

            Arc::new(OpenAiProvider {
                prompt,
                model,
                options: OpenAiOptions { max_output_tokens },
                pricing: pricing_from_config(cfg),
                client: reqwest::Client::new(),
            })
        }
        _ => {
            let prompt = cfg
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("{value}")
                .to_string();
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

// -----------------------------------------------------------------------------
fn now_iso() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn hash_atom(hasher: &mut Sha256, ctx: &AiContextAtom) {
    hasher.update(format!("\x00{}\x00{}", ctx.typ, ctx.value).as_bytes());
    if let Some(m) = ctx.mime.as_ref() {
        hasher.update(format!("\x00{}\x00", m).as_bytes());
    }
    hasher.update(ctx.meta.as_bytes());
}

fn hash_cache_key(model_config: &str, input: &ModelInput) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}\x00", CACHE_SCHEMA_VERSION).as_bytes());
    hasher.update(model_config.as_bytes());
    match input {
        ModelInput::Atom(ctx) => {
            hasher.update(b"\x00atom");
            hash_atom(&mut hasher, ctx);
        }
        ModelInput::Batch(batch) => {
            hasher.update(b"\x00batch");
            for item in &batch.items {
                hash_atom(&mut hasher, item);
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

const IMAGE_TOKEN_EST: u64 = 1024;

fn estimate_text_tokens(value: &str) -> u64 {
    let len = value.chars().count() as u64;
    ((len + 3) / 4).max(1)
}

fn estimate_tokens(ctx: &AiContext) -> u64 {
    if matches!(ctx.typ.as_str(), "image" | "image_url" | "image_path") {
        IMAGE_TOKEN_EST
    } else if ctx.value_is_null {
        0
    } else if ctx.typ == "null" {
        0
    } else {
        estimate_text_tokens(&ctx.value)
    }
}

fn estimate_model_input(input: &ModelInput) -> u64 {
    match input {
        ModelInput::Atom(ctx) => estimate_tokens(ctx),
        ModelInput::Batch(batch) => batch.items.iter().map(estimate_tokens).sum(),
    }
}

fn cache_entries_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("cache_entries.jsonl")
}

fn load_disk_cache(cache_dir: &Path) -> PolarsResult<HashMap<String, CacheDiskRow>> {
    let path = cache_entries_path(cache_dir);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let f = File::open(&path).map_err(|e| {
        PolarsError::ComputeError(format!("cache open {}: {}", path.display(), e).into())
    })?;
    let reader = BufReader::new(f);
    let mut map = HashMap::new();
    for line in reader.lines() {
        let line =
            line.map_err(|e| PolarsError::ComputeError(format!("cache read: {}", e).into()))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<CacheDiskRow>(&line) {
            map.insert(row.cache_key.clone(), row);
        }
    }
    Ok(map)
}

fn append_cache_entries(cache_dir: &Path, rows: &[CacheDiskRow]) -> PolarsResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(cache_dir).map_err(|e| {
        PolarsError::ComputeError(format!("cache mkdir {}: {}", cache_dir.display(), e).into())
    })?;
    let path = cache_entries_path(cache_dir);
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| {
            PolarsError::ComputeError(format!("cache append {}: {}", path.display(), e).into())
        })?;
    for r in rows {
        let js = serde_json::to_string(r)
            .map_err(|e| PolarsError::ComputeError(format!("cache serialize: {}", e).into()))?;
        writeln!(f, "{}", js)
            .map_err(|e| PolarsError::ComputeError(format!("cache write: {}", e).into()))?;
    }
    Ok(())
}

fn extract_context_rows(series: &Series) -> PolarsResult<Vec<AiContext>> {
    let struct_series = series.struct_().map_err(|_| {
        PolarsError::InvalidOperation(
            "`polars-ai` expects an AiModelContext Struct column (_type,_value,_mime,_meta)."
                .into(),
        )
    })?;
    let type_series = struct_series.field_by_name("_type")?;
    let value_series = struct_series.field_by_name("_value")?;
    let mime_series = struct_series.field_by_name("_mime")?;
    let meta_series = struct_series.field_by_name("_meta")?;

    let type_ca = type_series.str()?;
    let mime_ca = mime_series.str()?;
    let meta_ca = meta_series.str()?;
    let value_ca = value_series.str()?;

    let mut out = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        let raw_type = type_ca.get(i);
        let t = raw_type.unwrap_or("null");

        let v_opt = value_ca.get(i);
        let vnull = v_opt.is_none();
        let v = v_opt
            .map(|s| s.to_string())
            .unwrap_or_else(|| "null".to_string());

        let mime_raw = mime_ca.get(i).map(|x| x.to_string());
        let mime = mime_raw.and_then(|s| if s == "null" { None } else { Some(s) });

        let e = meta_ca.get(i).unwrap_or("{}");

        let value_is_null = vnull || v == "null";

        out.push(AiContext {
            typ: t.to_string(),
            value: if value_is_null { String::new() } else { v },
            mime,
            meta: e.to_string(),
            value_is_null,
        });
    }
    Ok(out)
}

fn extract_context_batches(series: &Series) -> PolarsResult<Vec<AiContextBatch>> {
    let mut out = Vec::with_capacity(series.len());

    for i in 0..series.len() {
        match series.get(i)? {
            AnyValue::List(list_series) => {
                out.push(AiContextBatch {
                    items: extract_context_rows(&list_series)?,
                });
            }
            AnyValue::Null => {
                out.push(AiContextBatch { items: Vec::new() });
            }
            other => {
                return Err(PolarsError::InvalidOperation(
                    format!(
                        "`polars-ai` expects a ContextBatch List[AiModelContext], got `{}`.",
                        other.dtype()
                    )
                    .into(),
                ));
            }
        }
    }

    Ok(out)
}

fn context_atom_dtype(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("ai_context"),
        DataType::Struct(vec![
            Field::new("_type".into(), DataType::String),
            Field::new("_value".into(), DataType::String),
            Field::new("_mime".into(), DataType::String),
            Field::new("_meta".into(), DataType::String),
        ]),
    ))
}

fn ai_response_dtype(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from_static("ai_response"),
        DataType::Struct(vec![
            Field::new("status".into(), DataType::String),
            Field::new("value".into(), DataType::String),
            Field::new("cache_key".into(), DataType::String),
            Field::new("model_config".into(), DataType::String),
            Field::new("error".into(), DataType::String),
            Field::new("attempts".into(), DataType::UInt32),
            Field::new("input_tokens".into(), DataType::UInt64),
            Field::new("output_tokens".into(), DataType::UInt64),
            Field::new("total_tokens".into(), DataType::UInt64),
            Field::new("cost_usd".into(), DataType::Float64),
            Field::new("created_at".into(), DataType::String),
            Field::new("completed_at".into(), DataType::String),
        ]),
    ))
}

fn assemble_context_series(ctxs: &[AiContextAtom]) -> PolarsResult<Series> {
    let n = ctxs.len();
    let typ: Vec<Option<&str>> = ctxs.iter().map(|c| Some(c.typ.as_str())).collect();
    let value: Vec<Option<&str>> = ctxs
        .iter()
        .map(|c| {
            if c.value_is_null {
                None
            } else {
                Some(c.value.as_str())
            }
        })
        .collect();
    let mime: Vec<Option<&str>> = ctxs.iter().map(|c| c.mime.as_deref()).collect();
    let meta: Vec<Option<&str>> = ctxs.iter().map(|c| Some(c.meta.as_str())).collect();
    let cols = vec![
        StringChunked::from_iter_options(PlSmallStr::from_static("_type"), typ.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("_value"), value.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("_mime"), mime.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("_meta"), meta.into_iter())
            .into_series(),
    ];

    StructChunked::from_series(PlSmallStr::from_static("ai_context"), n, cols.iter())
        .map(|s| s.into_series())
}

#[derive(Clone, Debug)]
struct ResponseRowParts {
    status: String,
    value: Option<String>,
    cache_key: String,
    model_config: String,
    error: Option<String>,
    attempts: u32,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cost_usd: Option<f64>,
    created_at: String,
    completed_at: String,
}

fn is_terminal(status: &str) -> bool {
    matches!(
        status,
        STAT_OK | STAT_CACHE_HIT | STAT_MODEL_ERROR | STAT_INVALID_CTX
    )
}

fn read_existing_parts(resp_series: &Series, idx: usize) -> PolarsResult<ResponseRowParts> {
    let ss = resp_series.struct_().map_err(|_| {
        PolarsError::InvalidOperation("`.ai.hydrate()` expects an AiResponse Struct column.".into())
    })?;
    let fst = ss.field_by_name("status")?;
    let fval = ss.field_by_name("value")?;
    let fk = ss.field_by_name("cache_key")?;
    let fmc = ss.field_by_name("model_config")?;
    let fer = ss.field_by_name("error")?;
    let fat = ss.field_by_name("attempts")?;
    let fit = ss.field_by_name("input_tokens")?;
    let fot = ss.field_by_name("output_tokens")?;
    let ftt = ss.field_by_name("total_tokens")?;
    let fcu = ss.field_by_name("cost_usd")?;
    let fcr = ss.field_by_name("created_at")?;
    let fco = ss.field_by_name("completed_at")?;

    let status_s = fst.str()?;
    let value_s = fval.str()?;
    let key_s = fk.str()?;
    let mc_s = fmc.str()?;
    let er_s = fer.str()?;
    let at_s = fat.u32()?;
    let it_s = fit.u64()?;
    let ot_s = fot.u64()?;
    let tt_s = ftt.u64()?;
    let cu_s = fcu.f64()?;
    let cre_s = fcr.str()?;
    let com_s = fco.str()?;

    Ok(ResponseRowParts {
        status: status_s.get(idx).unwrap_or("").to_string(),
        value: value_s.get(idx).map(|x| x.to_string()),
        cache_key: key_s.get(idx).unwrap_or("").to_string(),
        model_config: mc_s.get(idx).unwrap_or("").to_string(),
        error: er_s.get(idx).map(|x| x.to_string()),
        attempts: at_s.get(idx).unwrap_or(0),
        input_tokens: it_s.get(idx),
        output_tokens: ot_s.get(idx),
        total_tokens: tt_s.get(idx),
        cost_usd: cu_s.get(idx),
        created_at: cre_s.get(idx).unwrap_or("").to_string(),
        completed_at: com_s.get(idx).unwrap_or("").to_string(),
    })
}

fn assemble_struct_series(parts: &[ResponseRowParts]) -> PolarsResult<Series> {
    let n = parts.len();
    let statuses: Vec<Option<&str>> = parts.iter().map(|r| Some(r.status.as_str())).collect();
    let values: Vec<Option<&str>> = parts.iter().map(|r| r.value.as_deref()).collect();
    let keys: Vec<Option<&str>> = parts.iter().map(|r| Some(r.cache_key.as_str())).collect();
    let mcfg: Vec<Option<&str>> = parts
        .iter()
        .map(|r| Some(r.model_config.as_str()))
        .collect();
    let errors: Vec<Option<&str>> = parts.iter().map(|r| r.error.as_deref()).collect();
    let attempts: Vec<u32> = parts.iter().map(|r| r.attempts).collect();
    let input_tokens: Vec<Option<u64>> = parts.iter().map(|r| r.input_tokens).collect();
    let output_tokens: Vec<Option<u64>> = parts.iter().map(|r| r.output_tokens).collect();
    let total_tokens: Vec<Option<u64>> = parts.iter().map(|r| r.total_tokens).collect();
    let cost_usd: Vec<Option<f64>> = parts.iter().map(|r| r.cost_usd).collect();
    let cr: Vec<Option<&str>> = parts.iter().map(|r| Some(r.created_at.as_str())).collect();
    let co: Vec<Option<&str>> = parts
        .iter()
        .map(|r| {
            if r.completed_at.is_empty() {
                None
            } else {
                Some(r.completed_at.as_str())
            }
        })
        .collect();

    let cols = vec![
        StringChunked::from_iter_options(PlSmallStr::from_static("status"), statuses.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("value"), values.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("cache_key"), keys.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("model_config"), mcfg.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("error"), errors.into_iter())
            .into_series(),
        ChunkedArray::<UInt32Type>::from_iter_options(
            PlSmallStr::from_static("attempts"),
            attempts.into_iter().map(Some),
        )
        .into_series(),
        ChunkedArray::<UInt64Type>::from_iter_options(
            PlSmallStr::from_static("input_tokens"),
            input_tokens.into_iter(),
        )
        .into_series(),
        ChunkedArray::<UInt64Type>::from_iter_options(
            PlSmallStr::from_static("output_tokens"),
            output_tokens.into_iter(),
        )
        .into_series(),
        ChunkedArray::<UInt64Type>::from_iter_options(
            PlSmallStr::from_static("total_tokens"),
            total_tokens.into_iter(),
        )
        .into_series(),
        ChunkedArray::<Float64Type>::from_iter_options(
            PlSmallStr::from_static("cost_usd"),
            cost_usd.into_iter(),
        )
        .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("created_at"), cr.into_iter())
            .into_series(),
        StringChunked::from_iter_options(PlSmallStr::from_static("completed_at"), co.into_iter())
            .into_series(),
    ];

    StructChunked::from_series(PlSmallStr::from_static("ai_response"), n, cols.iter())
        .map(|s| s.into_series())
}

fn run_engine(
    inputs: &[ModelInput],
    prior: Option<&[ResponseRowParts]>,
    hydrate: bool,
    model_config: &str,
    kwargs: &MapKwargs,
) -> PolarsResult<Series> {
    let batch_started = now_iso();

    let rate = kwargs.rate_limit_per_second.unwrap_or(50);
    let limiter_opt = if rate == 0 {
        None
    } else {
        let q = Quota::per_second(NonZeroU32::new(rate.max(1)).unwrap());
        Some(Arc::new(RateLimiter::direct(q)))
    };

    let max_req_budget = kwargs.max_requests;
    let max_tok_budget = kwargs.max_tokens;
    let concurrency = kwargs.max_concurrency.unwrap_or(32).max(1);

    let mut disk_map = HashMap::new();
    let cache_enabled = kwargs.cache_enabled.unwrap_or(false);
    let cache_base = kwargs.cache_path.clone();
    let cache_dir = cache_base.filter(|_| cache_enabled).map(PathBuf::from);

    if let Some(ref dir) = cache_dir {
        disk_map = load_disk_cache(dir)?;
    }

    let n = inputs.len();
    let mut out: Vec<Option<ResponseRowParts>> = vec![None; n];

    let mut used_req: usize = 0;
    let mut used_tok: u64 = 0;
    let mut jobs: Vec<(usize, ModelInput, String)> = Vec::new();

    for i in 0..n {
        let input = &inputs[i];
        let key = hash_cache_key(model_config, input);

        if hydrate {
            let p_row = prior.and_then(|p| p.get(i)).ok_or_else(|| {
                PolarsError::ComputeError(
                    "`ai_hydrate` prior series length mismatches contexts.".into(),
                )
            })?;
            if is_terminal(&p_row.status) {
                out[i] = Some(p_row.clone());
                continue;
            }
        }

        let input_is_null = match input {
            ModelInput::Atom(ctx) => ctx.value_is_null,
            ModelInput::Batch(batch) => batch.items.iter().any(|item| item.value_is_null),
        };

        if input_is_null {
            out[i] = Some(ResponseRowParts {
                status: STAT_INVALID_CTX.into(),
                value: None,
                cache_key: key,
                model_config: model_config.to_string(),
                error: Some("null _value in AiModelContext".into()),
                attempts: 0,
                input_tokens: None,
                output_tokens: None,
                total_tokens: None,
                cost_usd: None,
                created_at: batch_started.clone(),
                completed_at: String::new(),
            });
            continue;
        }

        if let Some(hit) = disk_map.get(&key) {
            if hit.status == STAT_OK || hit.status == STAT_CACHE_HIT {
                out[i] = Some(ResponseRowParts {
                    status: STAT_CACHE_HIT.into(),
                    value: hit.value.clone(),
                    cache_key: key.clone(),
                    model_config: hit.model_config.clone(),
                    error: hit.error.clone(),
                    attempts: hit.attempts,
                    input_tokens: hit.input_tokens,
                    output_tokens: hit.output_tokens,
                    total_tokens: hit.total_tokens,
                    cost_usd: hit.cost_usd,
                    created_at: hit.created_at.clone(),
                    completed_at: hit.completed_at.clone(),
                });
                continue;
            }
        }

        let tok_need = estimate_model_input(input);
        let mut can_req = max_req_budget.map(|m| used_req < m).unwrap_or(true);
        let can_tok = max_tok_budget
            .map(|lim| used_tok.saturating_add(tok_need) <= lim)
            .unwrap_or(true);

        if !can_tok {
            can_req = false;
        }

        if can_req {
            used_req += 1;
            used_tok = used_tok.saturating_add(tok_need);
            jobs.push((i, input.clone(), key));
        } else {
            out[i] = Some(ResponseRowParts {
                status: STAT_BUDGET_EXHAUSTED.into(),
                value: None,
                cache_key: key,
                model_config: model_config.to_string(),
                error: None,
                attempts: 0,
                input_tokens: None,
                output_tokens: None,
                total_tokens: None,
                cost_usd: None,
                created_at: batch_started.clone(),
                completed_at: String::new(),
            });
        }
    }

    let cfg: serde_json::Value = serde_json::from_str(model_config).unwrap_or(Value::Null);
    let provider = provider_from_config(&cfg);
    let model_cfg_arc = Arc::new(model_config.to_string());
    let limiter_arc = limiter_opt;

    let mut new_cache_writes: Vec<CacheDiskRow> = Vec::new();

    let results_map: HashMap<usize, ResponseRowParts> = if jobs.is_empty() {
        HashMap::new()
    } else {
        let rt = Runtime::new()
            .map_err(|e| PolarsError::ComputeError(format!("tokio runtime: {}", e).into()))?;
        let sem = Arc::new(Semaphore::new(concurrency));
        let pairs: Vec<(usize, ResponseRowParts)> = rt.block_on(async {
            stream::iter(jobs.into_iter())
                .map(|(idx, ctx, ck)| {
                    let prov = Arc::clone(&provider);
                    let lim = limiter_arc.clone();
                    let sem = Arc::clone(&sem);
                    let mc = Arc::clone(&model_cfg_arc);
                    let batch_ts = batch_started.clone();
                    async move {
                        let _p = sem.acquire_owned().await.ok();
                        if let Some(l) = lim {
                            l.until_ready().await;
                        }
                        let res = prov.call(ctx).await;
                        let done_t = now_iso();
                        let row = match res {
                            Ok(output) => ResponseRowParts {
                                status: STAT_OK.into(),
                                value: Some(output.text),
                                cache_key: ck,
                                model_config: (*mc).clone(),
                                error: None,
                                attempts: 1,
                                input_tokens: output.input_tokens,
                                output_tokens: output.output_tokens,
                                total_tokens: output.total_tokens,
                                cost_usd: output.cost_usd,
                                created_at: batch_ts.clone(),
                                completed_at: done_t.clone(),
                            },
                            Err(e) => ResponseRowParts {
                                status: STAT_MODEL_ERROR.into(),
                                value: None,
                                cache_key: ck,
                                model_config: (*mc).clone(),
                                error: Some(e.0),
                                attempts: 1,
                                input_tokens: None,
                                output_tokens: None,
                                total_tokens: None,
                                cost_usd: None,
                                created_at: batch_ts,
                                completed_at: done_t,
                            },
                        };
                        (idx, row)
                    }
                })
                .buffer_unordered(concurrency)
                .collect()
                .await
        });
        pairs.into_iter().collect()
    };

    for (idx, row) in results_map {
        if cache_dir.is_some() && (row.status == STAT_OK || row.status == STAT_MODEL_ERROR) {
            new_cache_writes.push(CacheDiskRow {
                cache_key: row.cache_key.clone(),
                status: row.status.clone(),
                value: row.value.clone(),
                error: row.error.clone(),
                model_config: row.model_config.clone(),
                attempts: row.attempts,
                input_tokens: row.input_tokens,
                output_tokens: row.output_tokens,
                total_tokens: row.total_tokens,
                cost_usd: row.cost_usd,
                created_at: row.created_at.clone(),
                completed_at: row.completed_at.clone(),
            });
        }
        out[idx] = Some(row);
    }

    if let Some(ref dir) = cache_dir {
        append_cache_entries(dir, &new_cache_writes)?;
    }

    let mut final_rows: Vec<ResponseRowParts> = Vec::with_capacity(n);
    for slot in out {
        final_rows.push(
            slot.ok_or_else(|| PolarsError::ComputeError("internal: missing response row".into()))?,
        );
    }

    assemble_struct_series(&final_rows)
}

#[polars_expr(output_type=String)]
fn ai_cache_key(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let ctxs = extract_context_rows(series)?;
    let keys: Vec<String> = ctxs
        .iter()
        .map(|c| hash_cache_key(&kwargs.model_config, &ModelInput::Atom(c.clone())))
        .collect();
    let opts: Vec<Option<&str>> = keys.iter().map(|s| Some(s.as_str())).collect();
    Ok(
        StringChunked::from_iter_options(PlSmallStr::from_static("cache_key"), opts.into_iter())
            .into_series(),
    )
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_map(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let ctxs = extract_context_rows(series)?;
    let model_inputs: Vec<ModelInput> = ctxs.into_iter().map(ModelInput::Atom).collect();
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, None, false, &mc, &kwargs)
}

#[polars_expr(output_type=String)]
fn ai_batch_cache_key(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let batches = extract_context_batches(series)?;
    let keys: Vec<String> = batches
        .iter()
        .map(|batch| hash_cache_key(&kwargs.model_config, &ModelInput::Batch(batch.clone())))
        .collect();
    let opts: Vec<Option<&str>> = keys.iter().map(|s| Some(s.as_str())).collect();
    Ok(
        StringChunked::from_iter_options(PlSmallStr::from_static("cache_key"), opts.into_iter())
            .into_series(),
    )
}

#[polars_expr(output_type_func=context_atom_dtype)]
fn ai_reduce_text(inputs: &[Series], kwargs: ReduceTextKwargs) -> PolarsResult<Series> {
    let batches = extract_context_batches(&inputs[0])?;
    let ctxs: Vec<AiContextAtom> = batches
        .iter()
        .map(|batch| {
            reduce_batch_to_text_atom(
                batch,
                &kwargs.text_separator,
                kwargs.number_text_items,
            )
        })
        .collect();
    assemble_context_series(&ctxs)
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_batch_map(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let batches = extract_context_batches(&inputs[0])?;
    let multimodal = kwargs.multimodal.unwrap_or(true);
    let model_inputs: Vec<ModelInput> = if multimodal {
        batches.into_iter().map(ModelInput::Batch).collect()
    } else {
        batches
            .iter()
            .map(|batch| ModelInput::Atom(reduce_batch_to_text_atom(batch, "\n\n", false)))
            .collect()
    };
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, None, false, &mc, &kwargs)
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_hydrate(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    if inputs.len() != 2 {
        return Err(PolarsError::InvalidOperation(
            "`ai_hydrate` expects [existing_response, ctx]".into(),
        ));
    }
    let prior_series = &inputs[0];
    let ctx_series = &inputs[1];
    if prior_series.len() != ctx_series.len() {
        return Err(PolarsError::InvalidOperation(
            "`ai_hydrate` length mismatch between response and ctx".into(),
        ));
    }
    let ctxs = extract_context_rows(ctx_series)?;
    let model_inputs: Vec<ModelInput> = ctxs.into_iter().map(ModelInput::Atom).collect();
    let mut prior_parts = Vec::with_capacity(prior_series.len());
    for i in 0..prior_series.len() {
        prior_parts.push(read_existing_parts(prior_series, i)?);
    }
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, Some(&prior_parts), true, &mc, &kwargs)
}

#[pymodule]
fn _polars_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
