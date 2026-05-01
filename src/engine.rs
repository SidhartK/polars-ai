use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use polars::prelude::*;
use serde_json::Value;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::Semaphore;

use crate::cache::{
    append_cache_entries, estimate_model_input, hash_cache_key, load_disk_cache, CacheDiskRow,
};
use crate::providers::provider_from_config;
use crate::types::{
    assemble_struct_series, is_terminal, now_iso, MapKwargs, ModelInput, ResponseRowParts,
    STAT_BUDGET_EXHAUSTED, STAT_CACHE_HIT, STAT_INVALID_CTX, STAT_MODEL_ERROR, STAT_OK,
};

const RUN_STATE_TTL: Duration = Duration::from_secs(10 * 60);
const VERBOSE_LOG_INTERVAL: Duration = Duration::from_secs(2);
const VERBOSE_COMPLETION_INTERVAL: u64 = 10;

static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
static RUNS: OnceLock<Mutex<HashMap<String, Arc<RunState>>>> = OnceLock::new();
static LEGACY_RUN_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct Budget {
    used_req: usize,
    used_tok: u64,
}

#[derive(Debug, Default)]
struct PlanningCounts {
    rows_seen: u64,
    jobs_reserved: u64,
    cache_hits: u64,
    budget_exhausted: u64,
    invalid_context: u64,
    terminal_preserved: u64,
}

#[derive(Debug)]
struct Progress {
    rows_seen: u64,
    jobs_reserved: u64,
    completed: u64,
    ok: u64,
    errors: u64,
    cache_hits: u64,
    budget_exhausted: u64,
    invalid_context: u64,
    terminal_preserved: u64,
    started: Instant,
    last_log: Instant,
    last_logged_completed: u64,
}

impl Progress {
    fn new(now: Instant) -> Self {
        Self {
            rows_seen: 0,
            jobs_reserved: 0,
            completed: 0,
            ok: 0,
            errors: 0,
            cache_hits: 0,
            budget_exhausted: 0,
            invalid_context: 0,
            terminal_preserved: 0,
            started: now,
            last_log: now.checked_sub(VERBOSE_LOG_INTERVAL).unwrap_or(now),
            last_logged_completed: 0,
        }
    }
}

#[derive(Debug)]
struct RunState {
    budget: Mutex<Budget>,
    progress: Mutex<Progress>,
    semaphore: Arc<Semaphore>,
    limiter: Option<Arc<DefaultDirectRateLimiter>>,
    last_used: Mutex<Instant>,
}

impl RunState {
    fn new(concurrency: usize, rate: u32) -> Self {
        let limiter = if rate == 0 {
            None
        } else {
            let quota = Quota::per_second(NonZeroU32::new(rate.max(1)).unwrap());
            Some(Arc::new(RateLimiter::direct(quota)))
        };

        Self {
            budget: Mutex::new(Budget {
                used_req: 0,
                used_tok: 0,
            }),
            progress: Mutex::new(Progress::new(Instant::now())),
            semaphore: Arc::new(Semaphore::new(concurrency)),
            limiter,
            last_used: Mutex::new(Instant::now()),
        }
    }

    fn touch(&self) {
        *self.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    fn is_expired(&self, now: Instant) -> bool {
        let last_used = *self.last_used.lock().unwrap_or_else(|e| e.into_inner());
        now.duration_since(last_used) >= RUN_STATE_TTL
    }

    fn try_reserve(
        &self,
        max_requests: Option<usize>,
        max_tokens: Option<u64>,
        tok_need: u64,
    ) -> bool {
        let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());

        if max_requests.map(|m| budget.used_req >= m).unwrap_or(false) {
            return false;
        }
        if max_tokens
            .map(|lim| budget.used_tok.saturating_add(tok_need) > lim)
            .unwrap_or(false)
        {
            return false;
        }

        budget.used_req += 1;
        budget.used_tok = budget.used_tok.saturating_add(tok_need);
        true
    }

    fn record_planning(
        &self,
        counts: PlanningCounts,
        verbose: bool,
        label: &str,
        concurrency: usize,
    ) {
        let mut progress = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        let first_log = progress.rows_seen == 0 && progress.completed == 0;
        progress.rows_seen = progress.rows_seen.saturating_add(counts.rows_seen);
        progress.jobs_reserved = progress.jobs_reserved.saturating_add(counts.jobs_reserved);
        progress.cache_hits = progress.cache_hits.saturating_add(counts.cache_hits);
        progress.budget_exhausted = progress
            .budget_exhausted
            .saturating_add(counts.budget_exhausted);
        progress.invalid_context = progress
            .invalid_context
            .saturating_add(counts.invalid_context);
        progress.terminal_preserved = progress
            .terminal_preserved
            .saturating_add(counts.terminal_preserved);

        let now = Instant::now();
        let should_log = first_log || now.duration_since(progress.last_log) >= VERBOSE_LOG_INTERVAL;
        if verbose && should_log {
            self.log_progress_locked(&mut progress, label, concurrency, "planned");
        }
    }

    fn record_completion(&self, ok: bool, verbose: bool, label: &str, concurrency: usize) {
        let mut progress = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        progress.completed = progress.completed.saturating_add(1);
        if ok {
            progress.ok = progress.ok.saturating_add(1);
        } else {
            progress.errors = progress.errors.saturating_add(1);
        }

        let now = Instant::now();
        let completion_delta = progress
            .completed
            .saturating_sub(progress.last_logged_completed);
        let finished_known_work =
            progress.jobs_reserved > 0 && progress.completed >= progress.jobs_reserved;
        let should_log = completion_delta >= VERBOSE_COMPLETION_INTERVAL
            || now.duration_since(progress.last_log) >= VERBOSE_LOG_INTERVAL
            || finished_known_work;

        if verbose && should_log {
            self.log_progress_locked(&mut progress, label, concurrency, "progress");
        }
    }

    fn log_progress_locked(
        &self,
        progress: &mut Progress,
        label: &str,
        concurrency: usize,
        phase: &str,
    ) {
        let now = Instant::now();
        let elapsed = now.duration_since(progress.started).as_secs();
        let waiting = progress.jobs_reserved.saturating_sub(progress.completed);
        eprintln!(
            "[polars-ai {label}] {phase}: seen={} queued={} completed={} waiting={} ok={} errors={} cache_hits={} budget_exhausted={} invalid_context={} preserved={} concurrency={} elapsed={}s",
            progress.rows_seen,
            progress.jobs_reserved,
            progress.completed,
            waiting,
            progress.ok,
            progress.errors,
            progress.cache_hits,
            progress.budget_exhausted,
            progress.invalid_context,
            progress.terminal_preserved,
            concurrency,
            elapsed,
        );
        progress.last_log = now;
        progress.last_logged_completed = progress.completed;
    }
}

fn global_runtime() -> PolarsResult<&'static Runtime> {
    match RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())
    }) {
        Ok(rt) => Ok(rt),
        Err(e) => Err(PolarsError::ComputeError(
            format!("tokio runtime: {}", e).into(),
        )),
    }
}

fn legacy_run_id() -> String {
    let id = LEGACY_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("legacy-run-{}", id)
}

fn get_or_create_run_state(kwargs: &MapKwargs) -> Arc<RunState> {
    let run_id = kwargs.run_id.clone().unwrap_or_else(legacy_run_id);
    let concurrency = kwargs.max_concurrency.unwrap_or(32).max(1);
    let rate = kwargs.rate_limit_per_second.unwrap_or(50);
    let registry = RUNS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut runs = registry.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();

    runs.retain(|_, state| Arc::strong_count(state) > 1 || !state.is_expired(now));

    if let Some(state) = runs.get(&run_id) {
        state.touch();
        return Arc::clone(state);
    }

    let state = Arc::new(RunState::new(concurrency, rate));
    runs.insert(run_id, Arc::clone(&state));
    state
}

pub(crate) fn run_engine(
    inputs: &[ModelInput],
    prior: Option<&[ResponseRowParts]>,
    hydrate: bool,
    model_config: &str,
    kwargs: &MapKwargs,
) -> PolarsResult<Series> {
    let batch_started = now_iso();

    let max_req_budget = kwargs.max_requests;
    let max_tok_budget = kwargs.max_tokens;
    let concurrency = kwargs.max_concurrency.unwrap_or(32).max(1);
    let run_state = get_or_create_run_state(kwargs);

    let mut disk_map = HashMap::new();
    let cache_enabled = kwargs.cache_enabled.unwrap_or(false);
    let cache_base = kwargs.cache_path.clone();
    let cache_dir = cache_base.filter(|_| cache_enabled).map(PathBuf::from);

    if let Some(ref dir) = cache_dir {
        disk_map = load_disk_cache(dir)?;
    }

    let n = inputs.len();
    let mut out: Vec<Option<ResponseRowParts>> = vec![None; n];

    let mut jobs: Vec<(usize, ModelInput, String)> = Vec::new();
    let mut planning_counts = PlanningCounts::default();

    for i in 0..n {
        planning_counts.rows_seen = planning_counts.rows_seen.saturating_add(1);
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
                planning_counts.terminal_preserved =
                    planning_counts.terminal_preserved.saturating_add(1);
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
            planning_counts.invalid_context = planning_counts.invalid_context.saturating_add(1);
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
                planning_counts.cache_hits = planning_counts.cache_hits.saturating_add(1);
                continue;
            }
        }

        let tok_need = estimate_model_input(input);
        if run_state.try_reserve(max_req_budget, max_tok_budget, tok_need) {
            jobs.push((i, input.clone(), key));
            planning_counts.jobs_reserved = planning_counts.jobs_reserved.saturating_add(1);
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
            planning_counts.budget_exhausted = planning_counts.budget_exhausted.saturating_add(1);
        }
    }

    let label = if hydrate { "hydrate" } else { "infer" };
    run_state.record_planning(planning_counts, kwargs.verbose, label, concurrency);

    let cfg: serde_json::Value = serde_json::from_str(model_config).unwrap_or(Value::Null);
    let provider = provider_from_config(&cfg);
    let model_cfg_arc = Arc::new(model_config.to_string());

    let mut new_cache_writes: Vec<CacheDiskRow> = Vec::new();

    let results_map: HashMap<usize, ResponseRowParts> = if jobs.is_empty() {
        HashMap::new()
    } else {
        let rt = global_runtime()?;
        let run_state = Arc::clone(&run_state);
        let pairs: Vec<(usize, ResponseRowParts)> = rt.block_on(async {
            stream::iter(jobs.into_iter())
                .map(|(idx, ctx, ck)| {
                    let prov = Arc::clone(&provider);
                    let run_state = Arc::clone(&run_state);
                    let mc = Arc::clone(&model_cfg_arc);
                    let batch_ts = batch_started.clone();
                    let label = label;
                    let verbose = kwargs.verbose;
                    async move {
                        let semaphore = Arc::clone(&run_state.semaphore);
                        let _p = semaphore.acquire_owned().await.ok();
                        if let Some(l) = run_state.limiter.clone() {
                            l.until_ready().await;
                        }
                        let res = prov.call(ctx).await;
                        let done_t = now_iso();
                        let row = match res {
                            Ok(output) => {
                                run_state.record_completion(true, verbose, label, concurrency);
                                ResponseRowParts {
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
                                }
                            }
                            Err(e) => {
                                run_state.record_completion(false, verbose, label, concurrency);
                                ResponseRowParts {
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
                                }
                            }
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
