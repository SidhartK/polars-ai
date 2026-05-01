use chrono::{SecondsFormat, Utc};
use futures::future::BoxFuture;
use polars::prelude::*;
use serde::Deserialize;

pub(crate) const CACHE_KEY_SCHEMA_VERSION: &str = "3";
pub(crate) const STAT_OK: &str = "ok";
pub(crate) const STAT_CACHE_HIT: &str = "cache_hit";
pub(crate) const STAT_BUDGET_EXHAUSTED: &str = "budget_exhausted";
pub(crate) const STAT_MODEL_ERROR: &str = "model_error";
pub(crate) const STAT_INVALID_CTX: &str = "invalid_context";

// -----------------------------------------------------------------------------
// Plugin kwargs (JSON from Python register_plugin_function)
// -----------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct MapKwargs {
    #[serde(default)]
    pub(crate) run_id: Option<String>,
    #[serde(rename = "model_config")]
    pub(crate) model_config: String,
    #[serde(default)]
    pub(crate) max_requests: Option<usize>,
    #[serde(default)]
    pub(crate) max_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) max_concurrency: Option<usize>,
    #[serde(default)]
    pub(crate) cache_enabled: Option<bool>,
    #[serde(default)]
    pub(crate) cache_path: Option<String>,
    #[serde(default)]
    pub(crate) rate_limit_per_second: Option<u32>,
    #[serde(default)]
    pub(crate) multimodal: Option<bool>,
    #[serde(default = "default_input_type")]
    pub(crate) input_type: String,
    #[serde(default)]
    pub(crate) mime: Option<String>,
    #[serde(default = "default_text_separator")]
    pub(crate) text_separator: String,
    #[serde(default)]
    pub(crate) number_text_items: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub(crate) struct ReduceTextKwargs {
    #[serde(default = "default_text_separator")]
    pub(crate) text_separator: String,
    #[serde(default)]
    pub(crate) number_text_items: bool,
}

fn default_text_separator() -> String {
    "\n\n".to_string()
}

fn default_input_type() -> String {
    "text".to_string()
}
#[derive(Clone, Debug)]
pub(crate) struct AiContextAtom {
    pub(crate) typ: String,
    pub(crate) value: String,
    pub(crate) mime: Option<String>,
    #[allow(dead_code)]
    pub(crate) meta: String,
    pub(crate) value_is_null: bool,
}

pub(crate) type AiContext = AiContextAtom;

#[derive(Clone, Debug)]
pub(crate) struct InputBatch {
    pub(crate) items: Vec<AiContextAtom>,
}

#[derive(Clone, Debug)]
pub(crate) enum ModelInput {
    Atom(AiContextAtom),
    Batch(InputBatch),
}

#[derive(Debug)]
pub(crate) struct ModelError(pub(crate) String);

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelOutput {
    pub(crate) text: String,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) cost_usd: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct TokenPricing {
    pub(crate) input_per_million: f64,
    pub(crate) output_per_million: f64,
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

pub(crate) trait ModelProvider: Send + Sync {
    fn call(&self, input: ModelInput) -> BoxFuture<'static, Result<ModelOutput, ModelError>>;
}
pub(crate) fn now_iso() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

pub(crate) fn context_atom_dtype(_inputs: &[Field]) -> PolarsResult<Field> {
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

pub(crate) fn ai_response_dtype(_inputs: &[Field]) -> PolarsResult<Field> {
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

pub(crate) fn assemble_context_series(ctxs: &[AiContextAtom]) -> PolarsResult<Series> {
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
pub(crate) struct ResponseRowParts {
    pub(crate) status: String,
    pub(crate) value: Option<String>,
    pub(crate) cache_key: String,
    pub(crate) model_config: String,
    pub(crate) error: Option<String>,
    pub(crate) attempts: u32,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) cost_usd: Option<f64>,
    pub(crate) created_at: String,
    pub(crate) completed_at: String,
}

pub(crate) fn is_terminal(status: &str) -> bool {
    matches!(
        status,
        STAT_OK | STAT_CACHE_HIT | STAT_MODEL_ERROR | STAT_INVALID_CTX
    )
}

pub(crate) fn read_existing_parts(
    resp_series: &Series,
    idx: usize,
) -> PolarsResult<ResponseRowParts> {
    let ss = resp_series.struct_().map_err(|_| {
        PolarsError::InvalidOperation(
            "`pl_ai.hydrate(...)` expects an AiResponse Struct column.".into(),
        )
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

pub(crate) fn assemble_struct_series(parts: &[ResponseRowParts]) -> PolarsResult<Series> {
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
