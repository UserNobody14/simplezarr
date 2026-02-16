//! Zarr V2 array and group opening / loading.

use std::collections::HashMap;
use std::sync::Arc;

use crate::array::{
    ChunkGetterFn, CompressionInfo, UnifiedMetadata, UnifiedZarrArray, parse_chunk,
};
use crate::codecs::AnyCodec;
use crate::codecs::blosc::{BloscCname, BloscCodec, BloscShuffle};
use crate::codecs::bytes::BytesCodec;
use crate::codecs::gzip::GzipCodec;
use crate::codecs::lz4::Lz4Codec;
use crate::codecs::zlib::ZlibCodec;
use crate::codecs::zstd::ZstdCodec;
use crate::error::{ZarrError, ZarrResult};
use crate::group::{UnifiedGroupMetadata, UnifiedZarrGroup};
use crate::metadata::v2::{ZarrCompressor, ZarrConsolidatedMetadata, ZarrV2Metadata};
use crate::store::StorageBackend;

// ---------------------------------------------------------------------------
// Compressor -> codec list conversion
// ---------------------------------------------------------------------------

/// Convert a V2 compressor JSON object to a list of codecs, matching the
/// Haskell `zarrCompressorToAnyCodec` function.
pub fn compressor_to_codecs(comp: &ZarrCompressor) -> Vec<AnyCodec> {
    let id_lower = comp.id.to_lowercase();
    match id_lower.as_str() {
        "gzip" => {
            let level = get_config_int(&comp.config, "level").unwrap_or(5) as u32;
            vec![AnyCodec::Gzip(GzipCodec {
                level: level.min(9),
            })]
        }
        "blosc" => vec![AnyCodec::Blosc(blosc_codec_from_config(comp, None))],
        "zlib" => {
            let level = get_config_int(&comp.config, "level").unwrap_or(1) as u32;
            vec![AnyCodec::Zlib(ZlibCodec {
                level: level.min(9),
            })]
        }
        "lz4" => {
            let acc = get_config_int(&comp.config, "acceleration").unwrap_or(1) as i32;
            vec![AnyCodec::Lz4(Lz4Codec {
                acceleration: acc.clamp(0, 9),
            })]
        }
        "lz4hc" => vec![AnyCodec::Blosc(blosc_codec_from_config(
            comp,
            Some(BloscCname::Lz4hc),
        ))],
        "blosclz" => vec![AnyCodec::Blosc(blosc_codec_from_config(
            comp,
            Some(BloscCname::Blosclz),
        ))],
        "zstd" => {
            let level = get_config_int(&comp.config, "level").unwrap_or(5) as i32;
            vec![AnyCodec::Zstd(ZstdCodec {
                level: level.clamp(0, 9),
            })]
        }
        "snappy" => vec![AnyCodec::Blosc(blosc_codec_from_config(
            comp,
            Some(BloscCname::Snappy),
        ))],
        _ => vec![],
    }
}

fn blosc_codec_from_config(
    comp: &ZarrCompressor,
    fallback_cname: Option<BloscCname>,
) -> BloscCodec {
    let cname = comp
        .config
        .get("cname")
        .and_then(|v| v.as_str())
        .and_then(parse_blosc_cname)
        .or(fallback_cname)
        .unwrap_or(BloscCname::Zstd);

    let clevel = get_config_int(&comp.config, "clevel")
        .or_else(|| get_config_int(&comp.config, "level"))
        .unwrap_or(5) as i32;

    let shuffle = comp.config.get("shuffle").and_then(|v| {
        if let Some(n) = v.as_i64() {
            match n {
                0 => Some(BloscShuffle::NoShuffle),
                1 => Some(BloscShuffle::Shuffle),
                2 => Some(BloscShuffle::BitShuffle),
                _ => None,
            }
        } else if let Some(s) = v.as_str() {
            match s.to_lowercase().as_str() {
                "noshuffle" | "0" => Some(BloscShuffle::NoShuffle),
                "shuffle" | "1" => Some(BloscShuffle::Shuffle),
                "bitshuffle" | "2" => Some(BloscShuffle::BitShuffle),
                _ => None,
            }
        } else {
            None
        }
    });

    let blocksize = get_config_int(&comp.config, "blocksize").unwrap_or(0) as usize;

    BloscCodec {
        typesize: None,
        cname,
        clevel: clevel.clamp(0, 9),
        shuffle,
        blocksize,
    }
}

fn parse_blosc_cname(s: &str) -> Option<BloscCname> {
    match s.to_lowercase().as_str() {
        "lz4" => Some(BloscCname::Lz4),
        "lz4hc" => Some(BloscCname::Lz4hc),
        "blosclz" => Some(BloscCname::Blosclz),
        "zstd" => Some(BloscCname::Zstd),
        "snappy" => Some(BloscCname::Snappy),
        "zlib" => Some(BloscCname::Zlib),
        _ => None,
    }
}

fn get_config_int(config: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<i64> {
    config.get(key).and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))
    })
}

/// Build the full codec list for a V2 array (compressor codecs + endian bytes codec).
fn get_codec_equivalents(md: &ZarrV2Metadata) -> Vec<AnyCodec> {
    let mut codecs = match &md.compressor {
        Some(comp) => compressor_to_codecs(comp),
        None => vec![],
    };
    // Append a BytesCodec with the correct endianness
    codecs.push(AnyCodec::Bytes(BytesCodec::new(md.dtype.byte_order)));
    codecs
}

// ---------------------------------------------------------------------------
// Create chunk getter
// ---------------------------------------------------------------------------

fn create_v2_chunk_getter<S: StorageBackend + 'static>(
    store: Arc<S>,
    base_path: String,
    md: ZarrV2Metadata,
) -> ChunkGetterFn {
    let codecs = get_codec_equivalents(&md);
    let md = Arc::new(md);
    let codecs = Arc::new(codecs);

    Arc::new(move |key: Vec<usize>| {
        let store = store.clone();
        let base_path = base_path.clone();
        let md = md.clone();
        let codecs = codecs.clone();

        Box::pin(async move {
            if key.len() != md.shape.len() {
                return Err(ZarrError::Other(
                    "Key dimensionality must match array shape".into(),
                ));
            }

            let key_str: String = key
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(".");

            if !md.keys.contains(&key_str) {
                return Err(ZarrError::NotFound(format!(
                    "Storage key {key_str} not found"
                )));
            }

            let chunk_path = store.join(&base_path, &key_str);
            let bytes = store.get(&chunk_path).await?;

            let raw: Option<&[u8]> = bytes.as_deref();
            parse_chunk(raw, md.dtype.data_type, &md.chunks, &md.fill_value, &codecs).await
        })
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Open a Zarr V2 array, returning a `UnifiedZarrArray` ready for chunk access.
pub async fn open<S: StorageBackend + 'static>(
    store: Arc<S>,
    path: &str,
) -> ZarrResult<UnifiedZarrArray> {
    let zarray_path = store.join(path, ".zarray");
    let bytes = store
        .get(&zarray_path)
        .await?
        .ok_or_else(|| ZarrError::NotFound(format!("No .zarray at {path}")))?;

    let md = ZarrV2Metadata::parse(&bytes)?;

    let unified_md = UnifiedMetadata {
        shape: md.shape.clone(),
        chunk_shape: md.chunks.clone(),
        data_type: md.dtype.data_type,
        fill_value: md.fill_value.clone(),
        order: md.order,
        zarr_format: md.zarr_format,
        compression_info: CompressionInfo::V2Compression {
            compressor: md.compressor.clone(),
            filters: md.filters.clone(),
        },
        attributes: None,
        dimension_names: None,
        keys: md.keys.clone(),
    };

    Ok(UnifiedZarrArray {
        metadata: unified_md,
        store: store.clone(),
        path: path.to_string(),
        codecs: get_codec_equivalents(&md),
    })
}

/// Open a group of V2 arrays. Tries `.zmetadata` (consolidated) first,
/// falls back to opening each array individually.
pub async fn open_group<S: StorageBackend + 'static>(
    store: Arc<S>,
    path: &str,
    array_names: &[&str],
) -> ZarrResult<UnifiedZarrGroup> {
    let zmetadata_path = store.join(path, ".zmetadata");

    match store.get(&zmetadata_path).await? {
        Some(bytes) => {
            // Consolidated metadata
            let consolidated = ZarrConsolidatedMetadata::parse(&bytes)?;
            if consolidated.zarr_consolidated_format != 1 {
                return Err(ZarrError::Metadata(
                    "Metadata is not in zarr-consolidated-v1 format".into(),
                ));
            }

            let mut arrays = HashMap::new();
            for (name, md) in &consolidated.metadata {
                let unified_md = UnifiedMetadata {
                    shape: md.shape.clone(),
                    chunk_shape: md.chunks.clone(),
                    data_type: md.dtype.data_type,
                    fill_value: md.fill_value.clone(),
                    order: md.order,
                    zarr_format: md.zarr_format,
                    compression_info: CompressionInfo::V2Compression {
                        compressor: md.compressor.clone(),
                        filters: md.filters.clone(),
                    },
                    attributes: None,
                    dimension_names: None,
                    keys: md.keys.clone(),
                };

                let array_path = store.join(path, name);

                arrays.insert(
                    name.clone(),
                    UnifiedZarrArray {
                        metadata: unified_md,
                        store: store.clone(),
                        path: array_path,
                        codecs: get_codec_equivalents(&md),
                    },
                );
            }

            let group_md = UnifiedGroupMetadata {
                zarr_format: 2,
                attributes: None,
                consolidated: true,
                array_names: consolidated.metadata.keys().cloned().collect(),
                path: path.to_string(),
            };

            Ok(UnifiedZarrGroup {
                metadata: group_md,
                arrays,
            })
        }
        None => {
            // No consolidated metadata -- open arrays individually.
            let mut arrays = HashMap::new();
            let mut errors = Vec::new();

            let handles: Vec<_> = array_names
                .iter()
                .map(|name| {
                    let store = store.clone();
                    let array_path = store.join(path, name);
                    let name = name.to_string();
                    tokio::spawn(async move {
                        let result = open(store, &array_path).await;
                        (name, result)
                    })
                })
                .collect();

            for handle in handles {
                match handle.await {
                    Ok((name, Ok(array))) => {
                        arrays.insert(name, array);
                    }
                    Ok((_, Err(e))) => errors.push(e),
                    Err(e) => errors.push(ZarrError::Other(format!("Task join error: {e}"))),
                }
            }

            if let Some(err) = errors.into_iter().next() {
                return Err(err);
            }

            let group_md = UnifiedGroupMetadata {
                zarr_format: 2,
                attributes: None,
                consolidated: false,
                array_names: array_names.iter().map(|s| s.to_string()).collect(),
                path: path.to_string(),
            };

            Ok(UnifiedZarrGroup {
                metadata: group_md,
                arrays,
            })
        }
    }
}
