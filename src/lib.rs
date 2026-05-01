//! polars-ai: expression plugins for provider-backed inference and hydration.

mod cache;
mod engine;
mod input;
mod plugins;
mod pricing;
mod providers;
mod types;

use pyo3::prelude::*;

#[pymodule]
fn _polars_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
