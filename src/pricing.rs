use serde_json::Value;

use crate::types::TokenPricing;

pub(crate) fn number_from_config(cfg: &Value, key: &str) -> Option<f64> {
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

pub(crate) fn pricing_from_config(cfg: &Value) -> Option<TokenPricing> {
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
            .get("name")
            .or_else(|| cfg.get("model"))
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

pub(crate) fn calculate_cost_usd(
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    pricing: Option<&TokenPricing>,
) -> Option<f64> {
    let pricing = pricing?;
    let input = input_tokens? as f64;
    let output = output_tokens? as f64;
    Some((input * pricing.input_per_million + output * pricing.output_per_million) / 1_000_000.0)
}
