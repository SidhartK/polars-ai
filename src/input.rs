use polars::prelude::*;
use serde_json::json;

use crate::types::{AiContext, AiContextAtom, InputBatch, MapKwargs, ModelInput};

pub(crate) fn is_text_joinable(typ: &str) -> bool {
    matches!(typ, "text" | "json" | "blob" | "unknown")
}

pub(crate) fn reduce_batch_to_text_atom(
    batch: &InputBatch,
    text_separator: &str,
    number_text_items: bool,
) -> AiContextAtom {
    if batch.items.iter().any(|item| item.value_is_null) {
        return AiContextAtom {
            typ: "text".to_string(),
            value: String::new(),
            mime: None,
            meta: json!({"reduction": "text", "error": "null _value in input list"}).to_string(),
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
                "input list contains `{}`; use multimodal inference instead of text reduction",
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
fn extract_context_rows(series: &Series) -> PolarsResult<Vec<AiContext>> {
    let struct_series = series.struct_().map_err(|_| {
        PolarsError::InvalidOperation(
            "`polars-ai` expects String, List[String], or an internal input Struct.".into(),
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

pub(crate) fn extract_internal_batches(series: &Series) -> PolarsResult<Vec<InputBatch>> {
    let mut out = Vec::with_capacity(series.len());

    for i in 0..series.len() {
        match series.get(i)? {
            AnyValue::List(list_series) => {
                out.push(InputBatch {
                    items: extract_context_rows(&list_series)?,
                });
            }
            AnyValue::Null => {
                out.push(InputBatch { items: Vec::new() });
            }
            other => {
                return Err(PolarsError::InvalidOperation(
                    format!(
                        "`polars-ai` expects an internal input List[Struct], got `{}`.",
                        other.dtype()
                    )
                    .into(),
                ));
            }
        }
    }

    Ok(out)
}

fn atom_from_value(value: Option<&str>, input_type: &str, mime: Option<String>) -> AiContextAtom {
    let value_is_null = value.is_none();
    AiContextAtom {
        typ: input_type.to_string(),
        value: value.unwrap_or("").to_string(),
        mime,
        meta: json!({"source": "polars_expr"}).to_string(),
        value_is_null,
    }
}

fn extract_text_rows(series: &Series, kwargs: &MapKwargs) -> PolarsResult<Vec<AiContextAtom>> {
    let values = series.str().map_err(|_| {
        PolarsError::InvalidOperation(
            "`polars-ai` expects a String, List[String], or internal input Struct expression."
                .into(),
        )
    })?;
    Ok((0..series.len())
        .map(|i| atom_from_value(values.get(i), &kwargs.input_type, kwargs.mime.clone()))
        .collect())
}

fn extract_text_batches(series: &Series, kwargs: &MapKwargs) -> PolarsResult<Vec<InputBatch>> {
    let mut out = Vec::with_capacity(series.len());
    for i in 0..series.len() {
        match series.get(i)? {
            AnyValue::List(list_series) => {
                let values = list_series.str().map_err(|_| {
                    PolarsError::InvalidOperation(
                        "`polars-ai` expects grouped list inputs to contain String values.".into(),
                    )
                })?;
                let items = (0..list_series.len())
                    .map(|idx| {
                        atom_from_value(values.get(idx), &kwargs.input_type, kwargs.mime.clone())
                    })
                    .collect();
                out.push(InputBatch { items });
            }
            AnyValue::Null => out.push(InputBatch { items: Vec::new() }),
            other => {
                return Err(PolarsError::InvalidOperation(
                    format!(
                        "`polars-ai` expects grouped input as List[String], got `{}`.",
                        other.dtype()
                    )
                    .into(),
                ));
            }
        }
    }
    Ok(out)
}

pub(crate) fn extract_model_inputs(
    series: &Series,
    kwargs: &MapKwargs,
) -> PolarsResult<Vec<ModelInput>> {
    match series.dtype() {
        DataType::Struct(_) => Ok(extract_context_rows(series)?
            .into_iter()
            .map(ModelInput::Atom)
            .collect()),
        DataType::List(inner) => {
            let batches = match inner.as_ref() {
                DataType::Struct(_) => extract_internal_batches(series)?,
                DataType::String => extract_text_batches(series, kwargs)?,
                other => {
                    return Err(PolarsError::InvalidOperation(
                        format!(
                            "`polars-ai` expects List[String] or an internal List[Struct], got List[{}].",
                            other
                        )
                        .into(),
                    ));
                }
            };
            let multimodal = kwargs.multimodal.unwrap_or(true);
            let all_text_joinable = batches.iter().all(|batch| {
                batch
                    .items
                    .iter()
                    .all(|item| is_text_joinable(item.typ.as_str()))
            });
            if multimodal && !all_text_joinable {
                Ok(batches.into_iter().map(ModelInput::Batch).collect())
            } else {
                Ok(batches
                    .iter()
                    .map(|batch| {
                        ModelInput::Atom(reduce_batch_to_text_atom(
                            batch,
                            &kwargs.text_separator,
                            kwargs.number_text_items,
                        ))
                    })
                    .collect())
            }
        }
        DataType::String => Ok(extract_text_rows(series, kwargs)?
            .into_iter()
            .map(ModelInput::Atom)
            .collect()),
        other => Err(PolarsError::InvalidOperation(
            format!(
                "`polars-ai` expects String, List[String], or internal input Struct, got `{}`.",
                other
            )
            .into(),
        )),
    }
}
