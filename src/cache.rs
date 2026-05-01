use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::{AiContext, AiContextAtom, ModelInput, CACHE_KEY_SCHEMA_VERSION};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CacheDiskRow {
    pub(crate) cache_key: String,
    pub(crate) status: String,
    pub(crate) value: Option<String>,
    pub(crate) error: Option<String>,
    pub(crate) model_config: String,
    pub(crate) attempts: u32,
    #[serde(default)]
    pub(crate) input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) output_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) total_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) cost_usd: Option<f64>,
    pub(crate) created_at: String,
    pub(crate) completed_at: String,
}
fn hash_atom(hasher: &mut Sha256, ctx: &AiContextAtom) {
    hasher.update(format!("\x00{}\x00{}", ctx.typ, ctx.value).as_bytes());
    if let Some(m) = ctx.mime.as_ref() {
        hasher.update(format!("\x00{}\x00", m).as_bytes());
    }
    hasher.update(ctx.meta.as_bytes());
}

pub(crate) fn hash_cache_key(model_config: &str, input: &ModelInput) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}\x00", CACHE_KEY_SCHEMA_VERSION).as_bytes());
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

pub(crate) fn estimate_text_tokens(value: &str) -> u64 {
    let len = value.chars().count() as u64;
    ((len + 3) / 4).max(1)
}

pub(crate) fn estimate_tokens(ctx: &AiContext) -> u64 {
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

pub(crate) fn estimate_model_input(input: &ModelInput) -> u64 {
    match input {
        ModelInput::Atom(ctx) => estimate_tokens(ctx),
        ModelInput::Batch(batch) => batch.items.iter().map(estimate_tokens).sum(),
    }
}
fn cache_entries_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("cache_entries.jsonl")
}

pub(crate) fn load_disk_cache(cache_dir: &Path) -> PolarsResult<HashMap<String, CacheDiskRow>> {
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

pub(crate) fn append_cache_entries(cache_dir: &Path, rows: &[CacheDiskRow]) -> PolarsResult<()> {
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
