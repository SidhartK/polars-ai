use std::env;
use std::fs;
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::future::BoxFuture;
use serde_json::{json, Value};

use crate::cache::{estimate_model_input, estimate_text_tokens, estimate_tokens};
use crate::input::{is_text_joinable, reduce_batch_to_text_atom};
use crate::pricing::{calculate_cost_usd, number_from_config, pricing_from_config};
use crate::types::{
    AiContextAtom, InputBatch, ModelError, ModelInput, ModelOutput, ModelProvider, TokenPricing,
};

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
    temperature: Option<f64>,
}

#[derive(Clone, Debug)]
struct OpenAiProvider {
    prompt: String,
    model: String,
    options: OpenAiOptions,
    pricing: Option<TokenPricing>,
    client: reqwest::Client,
}

#[derive(Clone, Debug, Default)]
struct AnthropicOptions {
    max_output_tokens: Option<u64>,
    temperature: Option<f64>,
}

#[derive(Clone, Debug)]
struct AnthropicProvider {
    prompt: String,
    model: String,
    options: AnthropicOptions,
    pricing: Option<TokenPricing>,
    client: reqwest::Client,
}

#[derive(Clone, Debug, Default)]
struct GeminiOptions {
    max_output_tokens: Option<u64>,
    temperature: Option<f64>,
}

#[derive(Clone, Debug)]
struct GeminiProvider {
    prompt: String,
    model: String,
    options: GeminiOptions,
    pricing: Option<TokenPricing>,
    client: reqwest::Client,
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
                    "FakeModel does not support native multimodal input item `{}`",
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
        other => Err(ModelError(format!("unsupported input type `{}`", other))),
    }
}

fn render_openai_batch_content(batch: &InputBatch, prompt: &str) -> Result<Value, ModelError> {
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
                    "unsupported input type `{}` in input list",
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
            if let Some(temperature) = options.temperature {
                body["temperature"] = json!(temperature);
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

fn render_anthropic_content(input: &ModelInput, prompt: &str) -> Result<Vec<Value>, ModelError> {
    let atom = match input {
        ModelInput::Atom(ctx) => ctx.clone(),
        ModelInput::Batch(batch) => reduce_batch_to_text_atom(batch, "\n\n", false),
    };
    let rendered_prompt = prompt.replace("{value}", &atom.value);
    let mime = atom.mime.as_deref().unwrap_or("image/jpeg");
    match atom.typ.as_str() {
        "text" | "json" | "blob" | "unknown" => {
            Ok(vec![json!({"type": "text", "text": rendered_prompt})])
        }
        "image" => Ok(vec![
            json!({"type": "text", "text": rendered_prompt}),
            json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": atom.value}}),
        ]),
        "image_path" => {
            let bytes = fs::read(&atom.value).map_err(|e| {
                ModelError(format!("failed to read image path `{}`: {}", atom.value, e))
            })?;
            Ok(vec![
                json!({"type": "text", "text": rendered_prompt}),
                json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": BASE64_STANDARD.encode(bytes)}}),
            ])
        }
        "image_url" => Err(ModelError(
            "AnthropicModel requires base64 image or image_path inputs; image_url is not sent directly"
                .to_string(),
        )),
        other => Err(ModelError(format!("unsupported input type `{}`", other))),
    }
}

impl ModelProvider for AnthropicProvider {
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>> {
        let prompt = self.prompt.clone();
        let model = self.model.clone();
        let options = self.options.clone();
        let pricing = self.pricing.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let api_key = match env::var("ANTHROPIC_API_KEY") {
                Ok(key) if !key.trim().is_empty() => key,
                _ => return Err(ModelError("ANTHROPIC_API_KEY is not set".to_string())),
            };

            let content = render_anthropic_content(&input, &prompt)?;
            let mut body = json!({
                "model": model,
                "max_tokens": options.max_output_tokens.unwrap_or(1024),
                "messages": [{"role": "user", "content": content}],
            });
            if let Some(temperature) = options.temperature {
                body["temperature"] = json!(temperature);
            }

            let response = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
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
            let text = parsed
                .get("content")
                .and_then(|parts| parts.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ModelError("response did not include text content".to_string()))?;
            let usage = parsed.get("usage");
            let input_tokens = usage
                .and_then(|u| u.get("input_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| Some(estimate_model_input(&input)));
            let output_tokens = usage
                .and_then(|u| u.get("output_tokens"))
                .and_then(|v| v.as_u64())
                .or_else(|| Some(estimate_text_tokens(&text)));
            let total_tokens = match (input_tokens, output_tokens) {
                (Some(i), Some(o)) => Some(i.saturating_add(o)),
                _ => None,
            };
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

fn render_gemini_parts(input: &ModelInput, prompt: &str) -> Result<Vec<Value>, ModelError> {
    let atom = match input {
        ModelInput::Atom(ctx) => ctx.clone(),
        ModelInput::Batch(batch) => reduce_batch_to_text_atom(batch, "\n\n", false),
    };
    let rendered_prompt = prompt.replace("{value}", &atom.value);
    let mime = atom.mime.as_deref().unwrap_or("image/jpeg");
    match atom.typ.as_str() {
        "text" | "json" | "blob" | "unknown" => Ok(vec![json!({"text": rendered_prompt})]),
        "image" => Ok(vec![
            json!({"text": rendered_prompt}),
            json!({"inline_data": {"mime_type": mime, "data": atom.value}}),
        ]),
        "image_path" => {
            let bytes = fs::read(&atom.value).map_err(|e| {
                ModelError(format!("failed to read image path `{}`: {}", atom.value, e))
            })?;
            Ok(vec![
                json!({"text": rendered_prompt}),
                json!({"inline_data": {"mime_type": mime, "data": BASE64_STANDARD.encode(bytes)}}),
            ])
        }
        "image_url" => Ok(vec![
            json!({"text": rendered_prompt}),
            json!({"file_data": {"mime_type": mime, "file_uri": atom.value}}),
        ]),
        other => Err(ModelError(format!("unsupported input type `{}`", other))),
    }
}

impl ModelProvider for GeminiProvider {
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>> {
        let prompt = self.prompt.clone();
        let model = self.model.clone();
        let options = self.options.clone();
        let pricing = self.pricing.clone();
        let client = self.client.clone();

        Box::pin(async move {
            let api_key = env::var("GEMINI_API_KEY")
                .or_else(|_| env::var("GOOGLE_API_KEY"))
                .map_err(|_| {
                    ModelError("GEMINI_API_KEY or GOOGLE_API_KEY is not set".to_string())
                })?;
            if api_key.trim().is_empty() {
                return Err(ModelError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY is not set".to_string(),
                ));
            }
            let parts = render_gemini_parts(&input, &prompt)?;
            let mut body = json!({
                "contents": [{"role": "user", "parts": parts}],
            });
            let mut generation_config = json!({});
            if let Some(max_output_tokens) = options.max_output_tokens {
                generation_config["maxOutputTokens"] = json!(max_output_tokens);
            }
            if let Some(temperature) = options.temperature {
                generation_config["temperature"] = json!(temperature);
            }
            if generation_config
                .as_object()
                .map(|o| !o.is_empty())
                .unwrap_or(false)
            {
                body["generationConfig"] = generation_config;
            }
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );
            let response = client
                .post(url)
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
            let text = parsed
                .get("candidates")
                .and_then(|candidates| candidates.get(0))
                .and_then(|candidate| candidate.get("content"))
                .and_then(|content| content.get("parts"))
                .and_then(|parts| parts.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ModelError("response did not include candidate text".to_string()))?;
            let usage = parsed.get("usageMetadata");
            let input_tokens = usage
                .and_then(|u| u.get("promptTokenCount"))
                .and_then(|v| v.as_u64())
                .or_else(|| Some(estimate_model_input(&input)));
            let output_tokens = usage
                .and_then(|u| u.get("candidatesTokenCount"))
                .and_then(|v| v.as_u64())
                .or_else(|| Some(estimate_text_tokens(&text)));
            let total_tokens = usage
                .and_then(|u| u.get("totalTokenCount"))
                .and_then(|v| v.as_u64())
                .or_else(|| match (input_tokens, output_tokens) {
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
fn string_from_config(cfg: &Value, key: &str) -> Option<String> {
    cfg.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            cfg.get("options")
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn u64_from_config(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(|v| v.as_u64()).or_else(|| {
        cfg.get("options")
            .and_then(|o| o.get(key))
            .and_then(|v| v.as_u64())
    })
}

pub(crate) fn provider_from_config(cfg: &Value) -> Arc<dyn ModelProvider> {
    let provider = cfg
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("fake");

    match provider {
        "openai" => {
            let prompt = string_from_config(cfg, "prompt").unwrap_or_else(|| "{value}".to_string());
            let model = string_from_config(cfg, "name")
                .or_else(|| string_from_config(cfg, "model"))
                .unwrap_or_else(|| "gpt-4o-mini".to_string());
            let max_output_tokens = u64_from_config(cfg, "max_output_tokens")
                .or_else(|| u64_from_config(cfg, "max_tokens"));
            let temperature = number_from_config(cfg, "temperature");

            Arc::new(OpenAiProvider {
                prompt,
                model,
                options: OpenAiOptions {
                    max_output_tokens,
                    temperature,
                },
                pricing: pricing_from_config(cfg),
                client: reqwest::Client::new(),
            })
        }
        "anthropic" => {
            let prompt = string_from_config(cfg, "prompt").unwrap_or_else(|| "{value}".to_string());
            let model = string_from_config(cfg, "name")
                .or_else(|| string_from_config(cfg, "model"))
                .unwrap_or_else(|| "claude-3-5-haiku-latest".to_string());
            Arc::new(AnthropicProvider {
                prompt,
                model,
                options: AnthropicOptions {
                    max_output_tokens: u64_from_config(cfg, "max_output_tokens")
                        .or_else(|| u64_from_config(cfg, "max_tokens")),
                    temperature: number_from_config(cfg, "temperature"),
                },
                pricing: pricing_from_config(cfg),
                client: reqwest::Client::new(),
            })
        }
        "gemini" => {
            let prompt = string_from_config(cfg, "prompt").unwrap_or_else(|| "{value}".to_string());
            let model = string_from_config(cfg, "name")
                .or_else(|| string_from_config(cfg, "model"))
                .unwrap_or_else(|| "gemini-1.5-flash".to_string());
            Arc::new(GeminiProvider {
                prompt,
                model,
                options: GeminiOptions {
                    max_output_tokens: u64_from_config(cfg, "max_output_tokens")
                        .or_else(|| u64_from_config(cfg, "max_tokens")),
                    temperature: number_from_config(cfg, "temperature"),
                },
                pricing: pricing_from_config(cfg),
                client: reqwest::Client::new(),
            })
        }
        _ => {
            let prompt = string_from_config(cfg, "prompt").unwrap_or_else(|| "{value}".to_string());
            let tag = string_from_config(cfg, "tag").unwrap_or_else(|| "fake".to_string());

            Arc::new(FakeProvider { prompt, tag })
        }
    }
}
