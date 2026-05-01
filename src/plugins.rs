//! Polars plugin entrypoints.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::cache::hash_cache_key;
use crate::engine::run_engine;
use crate::input::{extract_internal_batches, extract_model_inputs, reduce_batch_to_text_atom};
use crate::types::{
    ai_response_dtype, assemble_context_series, context_atom_dtype, read_existing_parts,
    AiContextAtom, MapKwargs, ReduceTextKwargs,
};

#[polars_expr(output_type=String)]
fn ai_cache_key(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let model_inputs = extract_model_inputs(series, &kwargs)?;
    let keys: Vec<String> = model_inputs
        .iter()
        .map(|c| hash_cache_key(&kwargs.model_config, c))
        .collect();
    let opts: Vec<Option<&str>> = keys.iter().map(|s| Some(s.as_str())).collect();
    Ok(
        StringChunked::from_iter_options(PlSmallStr::from_static("cache_key"), opts.into_iter())
            .into_series(),
    )
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_infer(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let model_inputs = extract_model_inputs(&inputs[0], &kwargs)?;
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, None, false, &mc, &kwargs)
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_map(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let model_inputs = extract_model_inputs(&inputs[0], &kwargs)?;
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, None, false, &mc, &kwargs)
}

#[polars_expr(output_type=String)]
fn ai_batch_cache_key(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let model_inputs = extract_model_inputs(&inputs[0], &kwargs)?;
    let keys: Vec<String> = model_inputs
        .iter()
        .map(|input| hash_cache_key(&kwargs.model_config, input))
        .collect();
    let opts: Vec<Option<&str>> = keys.iter().map(|s| Some(s.as_str())).collect();
    Ok(
        StringChunked::from_iter_options(PlSmallStr::from_static("cache_key"), opts.into_iter())
            .into_series(),
    )
}

#[polars_expr(output_type_func=context_atom_dtype)]
fn ai_reduce_text(inputs: &[Series], kwargs: ReduceTextKwargs) -> PolarsResult<Series> {
    let batches = extract_internal_batches(&inputs[0])?;
    let ctxs: Vec<AiContextAtom> = batches
        .iter()
        .map(|batch| {
            reduce_batch_to_text_atom(batch, &kwargs.text_separator, kwargs.number_text_items)
        })
        .collect();
    assemble_context_series(&ctxs)
}

#[polars_expr(output_type_func=ai_response_dtype)]
fn ai_batch_map(inputs: &[Series], kwargs: MapKwargs) -> PolarsResult<Series> {
    let model_inputs = extract_model_inputs(&inputs[0], &kwargs)?;
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
    let model_inputs = extract_model_inputs(ctx_series, &kwargs)?;
    let mut prior_parts = Vec::with_capacity(prior_series.len());
    for i in 0..prior_series.len() {
        prior_parts.push(read_existing_parts(prior_series, i)?);
    }
    let mc = kwargs.model_config.clone();
    run_engine(&model_inputs, Some(&prior_parts), true, &mc, &kwargs)
}
