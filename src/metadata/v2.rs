use crate::error::{ZarrError, ZarrResult};
use crate::types::{ArrayOrder, DataType, Endian, FillValue};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// V2 DataType  (NumPy format wrapper)
// ---------------------------------------------------------------------------

/// V2-specific data type that wraps the core `DataType` along with byte order
/// and an optional time unit (for datetime/timedelta dtypes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2DataType {
    pub data_type: DataType,
    pub byte_order: Endian,
    pub time_unit: Option<String>,
}

/// Intermediate parsed representation of a NumPy format string.
#[derive(Debug)]
struct NumPyFormat {
    byte_order: char,
    type_code: char,
    byte_size: usize,
    time_unit: Option<String>,
}

/// Parse a NumPy dtype format string (e.g. `"<f8"`, `">i4"`, `"|b1"`, `"<M8[ns]"`)
/// into a [`V2DataType`].
pub fn parse_numpy_dtype(s: &str) -> Result<V2DataType, String> {
    let fmt = parse_numpy_format(s)?;
    numpy_format_to_dtype(&fmt)
}

fn parse_numpy_format(s: &str) -> Result<NumPyFormat, String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 3 {
        return Err(format!("NumPy format string too short: {s}"));
    }

    let byte_order = chars[0];
    if !['<', '>', '|'].contains(&byte_order) {
        return Err(format!("Invalid byte order: {byte_order}"));
    }

    let type_code = chars[1];
    if !['b', 'i', 'u', 'f', 'c', 'M', 'm', 'S', 'U', 'V'].contains(&type_code) {
        return Err(format!("Invalid type code: {type_code}"));
    }

    let rest: String = chars[2..].iter().collect();

    match type_code {
        'M' | 'm' => parse_with_time_unit(&rest).map(|(byte_size, time_unit)| NumPyFormat {
            byte_order,
            type_code,
            byte_size,
            time_unit,
        }),
        _ => {
            let byte_size: usize = rest
                .parse()
                .map_err(|_| format!("Invalid byte size: {rest}"))?;
            if byte_size == 0 {
                return Err(format!("Byte size must be > 0, got {rest}"));
            }
            Ok(NumPyFormat {
                byte_order,
                type_code,
                byte_size,
                time_unit: None,
            })
        }
    }
}

fn parse_with_time_unit(s: &str) -> Result<(usize, Option<String>), String> {
    if let Some(bracket_pos) = s.find('[') {
        let size_str = &s[..bracket_pos];
        let rest = &s[bracket_pos + 1..];
        let end = rest
            .find(']')
            .ok_or("Missing closing bracket in datetime format")?;
        let unit = rest[..end].to_string();
        let byte_size: usize = size_str
            .parse()
            .map_err(|_| format!("Invalid byte size in datetime format: {size_str}"))?;
        if byte_size == 0 {
            return Err("Byte size must be > 0".into());
        }
        Ok((byte_size, Some(unit)))
    } else {
        let byte_size: usize = s
            .parse()
            .map_err(|_| format!("Invalid byte size: {s}"))?;
        if byte_size == 0 {
            return Err("Byte size must be > 0".into());
        }
        Ok((byte_size, None))
    }
}

fn parse_byte_order(c: char) -> Result<Endian, String> {
    match c {
        '<' => Ok(Endian::Little),
        '>' => Ok(Endian::Big),
        '|' => Ok(Endian::NotApplicable),
        _ => Err(format!("Invalid byte order: {c}")),
    }
}

fn numpy_format_to_dtype(fmt: &NumPyFormat) -> Result<V2DataType, String> {
    let core = match (fmt.type_code, fmt.byte_size) {
        ('b', 1) => DataType::Bool,
        ('i', 1) => DataType::Int8,
        ('i', 2) => DataType::Int16,
        ('i', 4) => DataType::Int32,
        ('i', 8) => DataType::Int64,
        ('u', 1) => DataType::UInt8,
        ('u', 2) => DataType::UInt16,
        ('u', 4) => DataType::UInt32,
        ('u', 8) => DataType::UInt64,
        ('f', 2) => DataType::Float16,
        ('f', 4) => DataType::Float32,
        ('f', 8) => DataType::Float64,
        ('c', 8) => DataType::Complex64,
        ('c', 16) => DataType::Complex128,
        ('S', _) | ('U', _) => DataType::String,
        ('V', _) => DataType::Bytes,
        ('M', _) | ('m', _) => {
            // Treat datetime/timedelta as Int64 (epoch-based)
            DataType::Int64
        }
        _ => {
            return Err(format!(
                "Unsupported NumPy type: {}{}",
                fmt.type_code, fmt.byte_size
            ))
        }
    };

    Ok(V2DataType {
        data_type: core,
        byte_order: parse_byte_order(fmt.byte_order)?,
        time_unit: fmt.time_unit.clone(),
    })
}

// Serde: V2DataType serialises as the NumPy format string
impl Serialize for V2DataType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bo = match self.byte_order {
            Endian::Little => "<",
            Endian::Big => ">",
            Endian::NotApplicable => "|",
        };
        let (tc, bs) = match self.data_type {
            DataType::Bool => ("b", "1"),
            DataType::Int8 => ("i", "1"),
            DataType::Int16 => ("i", "2"),
            DataType::Int32 => ("i", "4"),
            DataType::Int64 => ("i", "8"),
            DataType::UInt8 => ("u", "1"),
            DataType::UInt16 => ("u", "2"),
            DataType::UInt32 => ("u", "4"),
            DataType::UInt64 => ("u", "8"),
            DataType::Float16 => ("f", "2"),
            DataType::Float32 => ("f", "4"),
            DataType::Float64 => ("f", "8"),
            DataType::Complex64 => ("c", "8"),
            DataType::Complex128 => ("c", "16"),
            DataType::String => ("S", "1"),
            DataType::Bytes => ("V", "1"),
        };
        let tu = self
            .time_unit
            .as_ref()
            .map(|u| format!("[{u}]"))
            .unwrap_or_default();
        serializer.serialize_str(&format!("{bo}{tc}{bs}{tu}"))
    }
}

impl<'de> Deserialize<'de> for V2DataType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        parse_numpy_dtype(&s).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// ZarrCompressor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZarrCompressor {
    pub id: String,
    #[serde(flatten)]
    pub config: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// ZarrV2Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ZarrV2Metadata {
    pub shape: Vec<usize>,
    pub chunks: Vec<usize>,
    pub dtype: V2DataType,

    #[serde(deserialize_with = "deserialize_fill_value_field")]
    pub fill_value: FillValue,

    #[serde(default = "default_order")]
    pub order: ArrayOrder,

    #[serde(default)]
    pub compressor: Option<ZarrCompressor>,

    #[serde(default)]
    pub filters: Option<serde_json::Value>,

    #[serde(default = "default_zarr_format", alias = "zarr_format")]
    pub zarr_format: u32,

    /// Computed storage keys (not from JSON; filled in after parsing).
    #[serde(skip)]
    pub keys: Vec<String>,
}

fn default_order() -> ArrayOrder {
    ArrayOrder::C
}

fn default_zarr_format() -> u32 {
    2
}

/// Custom serde for `ArrayOrder`
impl Serialize for ArrayOrder {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ArrayOrder::C => serializer.serialize_str("C"),
            ArrayOrder::F => serializer.serialize_str("F"),
        }
    }
}

impl<'de> Deserialize<'de> for ArrayOrder {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "C" | "c" => Ok(ArrayOrder::C),
            "F" | "f" => Ok(ArrayOrder::F),
            _ => Err(serde::de::Error::custom(format!("Unknown order: {s}"))),
        }
    }
}

/// Custom deserializer for `fill_value` that needs the sibling `dtype` field.
///
/// Since serde doesn't give us sibling access, we defer actual fill-value
/// interpretation to a post-parse step. Here we store the raw JSON value,
/// then resolve it via `ZarrV2Metadata::resolve_fill_value`.
fn deserialize_fill_value_field<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<FillValue, D::Error> {
    // Deserialize as raw JSON value; real parsing happens in post-processing.
    let _raw = serde_json::Value::deserialize(deserializer)?;
    // Placeholder -- will be resolved in post-parse.
    Ok(FillValue::NaN)
}

impl ZarrV2Metadata {
    /// Parse from raw JSON bytes, fully resolving fill_value and computing keys.
    pub fn parse(json_bytes: &[u8]) -> ZarrResult<Self> {
        // First parse to get the raw JSON
        let raw: serde_json::Value = serde_json::from_slice(json_bytes)
            .map_err(|e| ZarrError::Metadata(format!("Invalid JSON: {e}")))?;

        let obj = raw
            .as_object()
            .ok_or_else(|| ZarrError::Metadata("Expected JSON object".into()))?;

        // Parse dtype first so we can use it for fill_value
        let dtype_val = obj
            .get("dtype")
            .ok_or_else(|| ZarrError::Metadata("Missing 'dtype' field".into()))?;
        let dtype_str = dtype_val
            .as_str()
            .ok_or_else(|| ZarrError::Metadata("'dtype' must be a string".into()))?;
        let v2dtype =
            parse_numpy_dtype(dtype_str).map_err(ZarrError::Metadata)?;

        // Parse fill_value using the dtype
        let fill_val = obj.get("fill_value").unwrap_or(&serde_json::Value::Null);
        let fill_value = super::parse_fill_value(v2dtype.data_type, fill_val)
            .map_err(|e| ZarrError::Metadata(format!("fill_value: {e}")))?;

        // Parse the rest using serde
        let mut md: ZarrV2Metadata = serde_json::from_value(raw)
            .map_err(|e| ZarrError::Metadata(format!("Metadata parse error: {e}")))?;

        md.fill_value = fill_value;
        md.keys = list_keys(&md.shape, &md.chunks);
        Ok(md)
    }
}

// ---------------------------------------------------------------------------
// Consolidated metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ZarrConsolidatedMetadata {
    pub zarr_consolidated_format: u32,
    pub metadata: HashMap<String, ZarrV2Metadata>,
}

impl ZarrConsolidatedMetadata {
    /// Parse consolidated `.zmetadata` JSON.
    pub fn parse(json_bytes: &[u8]) -> ZarrResult<Self> {
        let raw: serde_json::Value = serde_json::from_slice(json_bytes)
            .map_err(|e| ZarrError::Metadata(format!("Invalid consolidated JSON: {e}")))?;

        let obj = raw
            .as_object()
            .ok_or_else(|| ZarrError::Metadata("Expected JSON object".into()))?;

        let format = obj
            .get("zarr_consolidated_format")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        let metadata_obj = obj
            .get("metadata")
            .and_then(|v| v.as_object())
            .ok_or_else(|| ZarrError::Metadata("Missing 'metadata' field".into()))?;

        let mut arrays = HashMap::new();

        for (key, value) in metadata_obj {
            // Filter out non-array keys
            if key.starts_with(".z") || key.ends_with(".zattrs") || key.ends_with(".zgroup") {
                continue;
            }
            // Only keep .zarray entries
            if !key.ends_with(".zarray") && !key.ends_with("/.zarray") {
                // Try parsing directly if it looks like array metadata
                if value.is_object() && value.get("shape").is_some() {
                    let json_bytes = serde_json::to_vec(value)
                        .map_err(|e| ZarrError::Metadata(format!("Re-serialize: {e}")))?;
                    match ZarrV2Metadata::parse(&json_bytes) {
                        Ok(md) => {
                            let name = key.replace("/.zarray", "").replace(".zarray", "");
                            let name = name.trim_start_matches('/').to_string();
                            arrays.insert(name, md);
                        }
                        Err(_) => continue,
                    }
                }
                continue;
            }

            let json_bytes = serde_json::to_vec(value)
                .map_err(|e| ZarrError::Metadata(format!("Re-serialize: {e}")))?;
            match ZarrV2Metadata::parse(&json_bytes) {
                Ok(md) => {
                    let name = key.replace("/.zarray", "").replace(".zarray", "");
                    let name = name.trim_start_matches('/').to_string();
                    arrays.insert(name, md);
                }
                Err(_) => continue,
            }
        }

        Ok(ZarrConsolidatedMetadata {
            zarr_consolidated_format: format,
            metadata: arrays,
        })
    }
}

// ---------------------------------------------------------------------------
// Key generation
// ---------------------------------------------------------------------------

/// Generate all storage keys for a given array shape and chunk sizes.
pub fn list_keys(shape: &[usize], chunks: &[usize]) -> Vec<String> {
    let chunks_per_dim: Vec<usize> = shape
        .iter()
        .zip(chunks.iter())
        .map(|(s, c)| (*s).div_ceil(*c))
        .collect();

    let all_indices = cartesian_product(&chunks_per_dim);
    all_indices
        .into_iter()
        .map(|idx| {
            idx.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(".")
        })
        .collect()
}

fn cartesian_product(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.is_empty() {
        return vec![vec![]];
    }
    let first = dims[0];
    let rest = cartesian_product(&dims[1..]);
    let mut result = Vec::new();
    for i in 0..first {
        for r in &rest {
            let mut v = vec![i];
            v.extend_from_slice(r);
            result.push(v);
        }
    }
    result
}
