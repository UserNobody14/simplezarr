use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::codecs::{apply_codec_pipeline, AnyCodec};
use crate::error::{ZarrError, ZarrResult};
use crate::types::{
    bytes_to_zarr_vector, fill_chunk, ArrayOrder, DataType, Endian, FillValue, ZarrValue,
    ZarrVectorValue,
};

// ---------------------------------------------------------------------------
// Internal chunk getter type
// ---------------------------------------------------------------------------

pub(crate) type ChunkGetterFn = Arc<
    dyn Fn(Vec<usize>) -> Pin<Box<dyn Future<Output = ZarrResult<ZarrVectorValue>> + Send>>
        + Send
        + Sync,
>;

// ---------------------------------------------------------------------------
// CompressionInfo
// ---------------------------------------------------------------------------

/// Unified compression / codec description (V2 or V3).
#[derive(Debug, Clone)]
pub enum CompressionInfo {
    V2Compression {
        compressor: Option<crate::metadata::v2::ZarrCompressor>,
        filters: Option<serde_json::Value>,
    },
    V3Codecs(Vec<AnyCodec>),
}

// ---------------------------------------------------------------------------
// UnifiedMetadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UnifiedMetadata {
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
    pub data_type: DataType,
    pub fill_value: FillValue,
    pub order: ArrayOrder,
    pub zarr_format: u32,
    pub compression_info: CompressionInfo,
    pub attributes: Option<serde_json::Map<String, serde_json::Value>>,
    pub dimension_names: Option<Vec<Option<String>>>,
    pub keys: Vec<String>,
}

// ---------------------------------------------------------------------------
// UnifiedZarrArray
// ---------------------------------------------------------------------------

pub struct UnifiedZarrArray {
    pub metadata: UnifiedMetadata,
    pub(crate) chunk_getter: ChunkGetterFn,
}

impl Clone for UnifiedZarrArray {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            chunk_getter: self.chunk_getter.clone(),
        }
    }
}

impl std::fmt::Debug for UnifiedZarrArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedZarrArray")
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl UnifiedZarrArray {
    /// Fetch a single chunk by its multi-dimensional indices.
    pub async fn get_chunk(&self, key: &[usize]) -> ZarrResult<ZarrVectorValue> {
        (self.chunk_getter)(key.to_vec()).await
    }

    /// Load all chunks concurrently and merge into a flat `Vec<f64>`.
    pub async fn load(&self) -> ZarrResult<Vec<f64>> {
        let keys = self.metadata.keys.clone();
        let getter = self.chunk_getter.clone();

        let handles: Vec<_> = keys
            .into_iter()
            .map(|key| {
                let getter = getter.clone();
                tokio::spawn(async move {
                    let indices = parse_key_string(&key);
                    let chunk = getter(indices).await?;
                    Ok::<_, ZarrError>((key, chunk))
                })
            })
            .collect();

        let mut chunk_map = HashMap::new();
        let mut errors = Vec::new();

        for handle in handles {
            match handle.await {
                Ok(Ok((key, chunk))) => {
                    chunk_map.insert(key, chunk);
                }
                Ok(Err(e)) => errors.push(e),
                Err(e) => errors.push(ZarrError::Other(format!("Task join error: {e}"))),
            }
        }

        if let Some(err) = errors.into_iter().next() {
            return Err(err);
        }

        merge_chunks(&chunk_map, &self.metadata)
    }

    /// Load all chunks concurrently and merge preserving element types.
    pub async fn load_value(&self) -> ZarrResult<ZarrVectorValue> {
        let keys = self.metadata.keys.clone();
        let getter = self.chunk_getter.clone();

        let handles: Vec<_> = keys
            .into_iter()
            .map(|key| {
                let getter = getter.clone();
                tokio::spawn(async move {
                    let indices = parse_key_string(&key);
                    let chunk = getter(indices).await?;
                    Ok::<_, ZarrError>((key, chunk))
                })
            })
            .collect();

        let mut chunk_map = HashMap::new();
        let mut errors = Vec::new();

        for handle in handles {
            match handle.await {
                Ok(Ok((key, chunk))) => {
                    chunk_map.insert(key, chunk);
                }
                Ok(Err(e)) => errors.push(e),
                Err(e) => errors.push(ZarrError::Other(format!("Task join error: {e}"))),
            }
        }

        if let Some(err) = errors.into_iter().next() {
            return Err(err);
        }

        merge_chunks_value(&chunk_map, &self.metadata)
    }
}

// ---------------------------------------------------------------------------
// Index math
// ---------------------------------------------------------------------------

/// Calculate strides for an N-dimensional array.
pub fn strides(shape: &[usize], order: ArrayOrder) -> Vec<usize> {
    match order {
        ArrayOrder::C => {
            // Row-major: last dimension varies fastest.
            let mut s: Vec<usize> = shape.iter().rev().scan(1usize, |state, &dim| {
                let stride = *state;
                *state *= dim;
                Some(stride)
            }).collect();
            s.reverse();
            s
        }
        ArrayOrder::F => {
            // Column-major: first dimension varies fastest.
            shape.iter().scan(1usize, |state, &dim| {
                let stride = *state;
                *state *= dim;
                Some(stride)
            }).collect()
        }
    }
}

/// Convert multi-dimensional indices to a flat linear index.
pub fn linear_index(shape: &[usize], order: ArrayOrder, indices: &[usize]) -> usize {
    let s = strides(shape, order);
    indices.iter().zip(s.iter()).map(|(i, s)| i * s).sum()
}

/// Generate all multi-dimensional index tuples within the given shape.
pub fn cartesian_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.is_empty() {
        return vec![vec![]];
    }
    let first = shape[0];
    let rest = cartesian_indices(&shape[1..]);
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

/// Parse a key string like `"0.1.2"` or `"0/1/2"` into indices.
pub fn parse_key_string(key: &str) -> Vec<usize> {
    let sep = if key.contains('.') { '.' } else { '/' };
    key.split(sep)
        .filter_map(|s| s.parse::<usize>().ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Chunk parsing
// ---------------------------------------------------------------------------

/// Parse a single chunk: decompress via codec pipeline, then interpret bytes.
pub async fn parse_chunk(
    data: Option<&[u8]>,
    dtype: DataType,
    chunk_shape: &[usize],
    fill_value: &FillValue,
    codecs: &[AnyCodec],
) -> ZarrResult<ZarrVectorValue> {
    match data {
        Some(raw) if !raw.is_empty() => {
            let decompressed = apply_codec_pipeline(codecs, raw).await?;

            // Determine endianness from the BytesCodec in the pipeline
            let endian = codecs
                .iter()
                .find_map(|c| c.bytes_endian())
                .unwrap_or(Endian::Little);

            bytes_to_zarr_vector(endian, dtype, &decompressed)
        }
        _ => {
            // Missing or empty chunk -> fill with fill value
            let scalar = fill_value.to_zarr_value(dtype);
            Ok(fill_chunk(&scalar, chunk_shape))
        }
    }
}

// ---------------------------------------------------------------------------
// Merge chunks into a flat array
// ---------------------------------------------------------------------------

/// Merge decoded chunks into a single flat `Vec<f64>`.
pub fn merge_chunks(
    chunk_map: &HashMap<String, ZarrVectorValue>,
    metadata: &UnifiedMetadata,
) -> ZarrResult<Vec<f64>> {
    let total_size: usize = metadata.shape.iter().product();
    let fill_f64 = metadata.fill_value.to_f64();
    let mut result = vec![fill_f64; total_size];
    let arr_strides = strides(&metadata.shape, metadata.order);

    for (key, chunk) in chunk_map {
        let key_indices = parse_key_string(key);
        let chunk_data = chunk.to_f64_vec()?;
        let chunk_indices = cartesian_indices(&metadata.chunk_shape);

        for (local_idx, local_pos) in chunk_indices.iter().enumerate() {
            // Global position = chunk_key * chunk_shape + local_pos
            let global: Vec<usize> = local_pos
                .iter()
                .zip(key_indices.iter())
                .zip(metadata.chunk_shape.iter())
                .map(|((lp, ki), cs)| ki * cs + lp)
                .collect();

            // Bounds check
            let in_bounds = global
                .iter()
                .zip(metadata.shape.iter())
                .all(|(g, s)| *g < *s);

            if in_bounds {
                let flat: usize = global.iter().zip(arr_strides.iter()).map(|(g, s)| g * s).sum();
                if flat < total_size && local_idx < chunk_data.len() {
                    result[flat] = chunk_data[local_idx];
                }
            }
        }
    }

    Ok(result)
}

/// Merge decoded chunks into a single `ZarrVectorValue::VWithNulls`,
/// preserving element types without lossy f64 conversion.
pub fn merge_chunks_value(
    chunk_map: &HashMap<String, ZarrVectorValue>,
    metadata: &UnifiedMetadata,
) -> ZarrResult<ZarrVectorValue> {
    let total_size: usize = metadata.shape.iter().product();
    let fill_scalar = metadata.fill_value.to_zarr_value(metadata.data_type);
    let mut result: Vec<Option<ZarrValue>> = vec![Some(fill_scalar); total_size];
    let arr_strides = strides(&metadata.shape, metadata.order);

    for (key, chunk) in chunk_map {
        let key_indices = parse_key_string(key);
        let chunk_vals = chunk.to_maybe_values();
        let chunk_indices = cartesian_indices(&metadata.chunk_shape);

        for (local_idx, local_pos) in chunk_indices.iter().enumerate() {
            let global: Vec<usize> = local_pos
                .iter()
                .zip(key_indices.iter())
                .zip(metadata.chunk_shape.iter())
                .map(|((lp, ki), cs)| ki * cs + lp)
                .collect();

            let in_bounds = global
                .iter()
                .zip(metadata.shape.iter())
                .all(|(g, s)| *g < *s);

            if in_bounds {
                let flat: usize = global.iter().zip(arr_strides.iter()).map(|(g, s)| g * s).sum();
                if flat < total_size && local_idx < chunk_vals.len() {
                    result[flat] = chunk_vals[local_idx].clone();
                }
            }
        }
    }

    Ok(ZarrVectorValue::VWithNulls(metadata.data_type, result))
}

