pub mod v2;

use crate::types::{DataType, FillValue, ZarrValue};
use half::f16;
use num_complex::Complex;

/// Parse a fill value from a JSON value, given the target data type.
/// Handles special string values like "NaN", "Infinity", "-Infinity",
/// JSON null, and normal numeric/bool/string values.
pub fn parse_fill_value(dtype: DataType, value: &serde_json::Value) -> Result<FillValue, String> {
    match value {
        serde_json::Value::Null => Ok(FillValue::NaN),

        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => match dtype {
                DataType::Float16 | DataType::Float32 | DataType::Float64
                | DataType::Complex64 | DataType::Complex128 => Ok(FillValue::NaN),
                _ => Err(format!("NaN not valid for {dtype:?}")),
            },
            "Infinity" => match dtype {
                DataType::Float16 | DataType::Float32 | DataType::Float64
                | DataType::Complex64 | DataType::Complex128 => Ok(FillValue::Infinity),
                _ => Err(format!("Infinity not valid for {dtype:?}")),
            },
            "-Infinity" => match dtype {
                DataType::Float16 | DataType::Float32 | DataType::Float64
                | DataType::Complex64 | DataType::Complex128 => Ok(FillValue::NegativeInfinity),
                _ => Err(format!("-Infinity not valid for {dtype:?}")),
            },
            _ => match dtype {
                DataType::String => Ok(FillValue::Value(ZarrValue::String(s.clone()))),
                DataType::Bytes => Ok(FillValue::Value(ZarrValue::Bytes(s.as_bytes().to_vec()))),
                _ => Err(format!(
                    "Expected {dtype:?} value, got string: {s}"
                )),
            },
        },

        serde_json::Value::Bool(b) => match dtype {
            DataType::Bool => Ok(FillValue::Value(ZarrValue::Bool(*b))),
            _ => Err(format!("Expected {dtype:?}, got bool")),
        },

        serde_json::Value::Number(n) => parse_numeric_fill(dtype, n),

        _ => Err(format!("Unexpected fill_value JSON: {value}")),
    }
}

fn parse_numeric_fill(dtype: DataType, n: &serde_json::Number) -> Result<FillValue, String> {
    match dtype {
        DataType::Int8 => {
            let i = n
                .as_i64()
                .ok_or_else(|| format!("Expected int for Int8, got {n}"))?;
            let v = i8::try_from(i).map_err(|_| format!("Value {i} out of range for Int8"))?;
            Ok(FillValue::Value(ZarrValue::Int8(v)))
        }
        DataType::Int16 => {
            let i = n.as_i64().ok_or_else(|| format!("Expected int for Int16, got {n}"))?;
            let v = i16::try_from(i).map_err(|_| format!("Value {i} out of range for Int16"))?;
            Ok(FillValue::Value(ZarrValue::Int16(v)))
        }
        DataType::Int32 => {
            let i = n.as_i64().ok_or_else(|| format!("Expected int for Int32, got {n}"))?;
            let v = i32::try_from(i).map_err(|_| format!("Value {i} out of range for Int32"))?;
            Ok(FillValue::Value(ZarrValue::Int32(v)))
        }
        DataType::Int64 => {
            let i = n.as_i64().ok_or_else(|| format!("Expected int for Int64, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Int64(i)))
        }
        DataType::UInt8 => {
            let i = n.as_u64().ok_or_else(|| format!("Expected uint for UInt8, got {n}"))?;
            let v = u8::try_from(i).map_err(|_| format!("Value {i} out of range for UInt8"))?;
            Ok(FillValue::Value(ZarrValue::UInt8(v)))
        }
        DataType::UInt16 => {
            let i = n.as_u64().ok_or_else(|| format!("Expected uint for UInt16, got {n}"))?;
            let v = u16::try_from(i).map_err(|_| format!("Value {i} out of range for UInt16"))?;
            Ok(FillValue::Value(ZarrValue::UInt16(v)))
        }
        DataType::UInt32 => {
            let i = n.as_u64().ok_or_else(|| format!("Expected uint for UInt32, got {n}"))?;
            let v = u32::try_from(i).map_err(|_| format!("Value {i} out of range for UInt32"))?;
            Ok(FillValue::Value(ZarrValue::UInt32(v)))
        }
        DataType::UInt64 => {
            let i = n.as_u64().ok_or_else(|| format!("Expected uint for UInt64, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::UInt64(i)))
        }
        DataType::Float16 => {
            let f = n.as_f64().ok_or_else(|| format!("Expected float for Float16, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Float16(f16::from_f64(f))))
        }
        DataType::Float32 => {
            let f = n.as_f64().ok_or_else(|| format!("Expected float for Float32, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Float32(f as f32)))
        }
        DataType::Float64 => {
            let f = n.as_f64().ok_or_else(|| format!("Expected float for Float64, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Float64(f)))
        }
        DataType::Complex64 => {
            let f = n.as_f64().ok_or_else(|| format!("Expected float for Complex64, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Complex64(Complex::new(
                f as f32, 0.0,
            ))))
        }
        DataType::Complex128 => {
            let f = n.as_f64().ok_or_else(|| format!("Expected float for Complex128, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Complex128(Complex::new(
                f, 0.0,
            ))))
        }
        DataType::Bool => {
            let i = n.as_i64().ok_or_else(|| format!("Expected int for Bool, got {n}"))?;
            Ok(FillValue::Value(ZarrValue::Bool(i != 0)))
        }
        DataType::String | DataType::Bytes => {
            Err(format!("Expected string for {dtype:?}, got number"))
        }
    }
}
