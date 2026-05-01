use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use futures::stream::{self, StreamExt};
use governor::{Quota, RateLimiter};
use polars::prelude::*;
use serde_json::Value;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

use crate::cache::{
    append_cache_entries, estimate_model_input, hash_cache_key, load_disk_cache, CacheDiskRow,
};
use crate::providers::provider_from_config;
use crate::types::{
    assemble_struct_series, is_terminal, now_iso, MapKwargs, ModelInput, ResponseRowParts,
    STAT_BUDGET_EXHAUSTED, STAT_CACHE_HIT, STAT_INVALID_CTX, STAT_MODEL_ERROR, STAT_OK,
};

pub(crate) fn run_engine(
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
                error: Some("null model input value".into()),
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
