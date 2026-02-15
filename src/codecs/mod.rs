pub mod blosc;
pub mod bytes;
pub mod fixedscaleoffset;
pub mod gzip;
pub mod lz4;
pub mod sharding;
pub mod zlib;
pub mod zstd;

use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CodecId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecId {
    Bytes,
    Gzip,
    Blosc,
    Zlib,
    Zstd,
    Lz4,
    Sharding,
    FixedScaleOffset,
}

impl std::fmt::Display for CodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecId::Bytes => write!(f, "bytes"),
            CodecId::Gzip => write!(f, "gzip"),
            CodecId::Blosc => write!(f, "blosc"),
            CodecId::Zlib => write!(f, "zlib"),
            CodecId::Zstd => write!(f, "zstd"),
            CodecId::Lz4 => write!(f, "lz4"),
            CodecId::Sharding => write!(f, "sharding_indexed"),
            CodecId::FixedScaleOffset => write!(f, "numcodecs.fixedscaleoffset"),
        }
    }
}

// ---------------------------------------------------------------------------
// AnyCodec  (enum dispatch, no Box<dyn>)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum AnyCodec {
    Bytes(bytes::BytesCodec),
    Gzip(gzip::GzipCodec),
    Blosc(blosc::BloscCodec),
    Zlib(zlib::ZlibCodec),
    Zstd(zstd::ZstdCodec),
    Lz4(lz4::Lz4Codec),
    Sharding(sharding::ShardingCodec),
    FixedScaleOffset(fixedscaleoffset::FixedScaleOffsetCodec),
}

impl AnyCodec {
    pub fn codec_id(&self) -> CodecId {
        match self {
            AnyCodec::Bytes(_) => CodecId::Bytes,
            AnyCodec::Gzip(_) => CodecId::Gzip,
            AnyCodec::Blosc(_) => CodecId::Blosc,
            AnyCodec::Zlib(_) => CodecId::Zlib,
            AnyCodec::Zstd(_) => CodecId::Zstd,
            AnyCodec::Lz4(_) => CodecId::Lz4,
            AnyCodec::Sharding(_) => CodecId::Sharding,
            AnyCodec::FixedScaleOffset(_) => CodecId::FixedScaleOffset,
        }
    }

    /// Decode bytes using this codec.
    pub async fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        match self {
            AnyCodec::Bytes(c) => c.decode(data),
            AnyCodec::Gzip(c) => c.decode(data),
            AnyCodec::Blosc(c) => c.decode(data).await,
            AnyCodec::Zlib(c) => c.decode(data),
            AnyCodec::Zstd(c) => c.decode(data),
            AnyCodec::Lz4(c) => c.decode(data),
            AnyCodec::Sharding(_) => Err(ZarrError::Codec(
                "Sharding codec decoding requires additional context".into(),
            )),
            AnyCodec::FixedScaleOffset(c) => c.decode(data),
        }
    }

    /// Encode bytes using this codec.
    pub async fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        match self {
            AnyCodec::Bytes(c) => c.encode(data),
            AnyCodec::Gzip(c) => c.encode(data),
            AnyCodec::Blosc(c) => c.encode(data).await,
            AnyCodec::Zlib(c) => c.encode(data),
            AnyCodec::Zstd(c) => c.encode(data),
            AnyCodec::Lz4(c) => c.encode(data),
            AnyCodec::Sharding(_) => Err(ZarrError::Codec(
                "Sharding codec encoding requires additional context".into(),
            )),
            AnyCodec::FixedScaleOffset(c) => c.encode(data),
        }
    }

    /// Get the bytes-codec endian config, if this is a BytesCodec.
    pub fn bytes_endian(&self) -> Option<crate::types::Endian> {
        match self {
            AnyCodec::Bytes(c) => c.endian,
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Codec pipeline
// ---------------------------------------------------------------------------

/// Apply a list of codecs to decode data. Codecs are applied in *reverse* order
/// (last codec decodes first), matching the Zarr spec.
pub async fn apply_codec_pipeline(codecs: &[AnyCodec], data: &[u8]) -> ZarrResult<Vec<u8>> {
    let mut buf = data.to_vec();
    for codec in codecs.iter().rev() {
        buf = codec.decode(&buf).await?;
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// JSON-based codec parsing  (V3 style)
// ---------------------------------------------------------------------------

/// JSON envelope that the V3 spec uses for codec entries.
#[derive(Debug, Deserialize, Serialize)]
struct CodecEnvelope {
    name: String,
    #[serde(default)]
    configuration: Option<serde_json::Value>,
}

/// Map a codec name string to its [`CodecId`].
pub fn lookup_codec_id(name: &str) -> Option<CodecId> {
    match name {
        "bytes" => Some(CodecId::Bytes),
        "gzip" => Some(CodecId::Gzip),
        "blosc" => Some(CodecId::Blosc),
        "zlib" => Some(CodecId::Zlib),
        "zstd" => Some(CodecId::Zstd),
        "lz4" => Some(CodecId::Lz4),
        "sharding_indexed" => Some(CodecId::Sharding),
        "numcodecs.fixedscaleoffset" => Some(CodecId::FixedScaleOffset),
        _ => None,
    }
}

/// Parse a single codec from a JSON value (V3 `{ "name": ..., "configuration": ... }` format).
pub fn parse_codec(value: &serde_json::Value) -> ZarrResult<AnyCodec> {
    let env: CodecEnvelope = serde_json::from_value(value.clone())
        .map_err(|e| ZarrError::Codec(format!("Invalid codec envelope: {e}")))?;

    let config = env.configuration.unwrap_or(serde_json::Value::Object(Default::default()));

    match lookup_codec_id(&env.name) {
        Some(CodecId::Bytes) => {
            let c: bytes::BytesCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| bytes::BytesCodec::default());
            Ok(AnyCodec::Bytes(c))
        }
        Some(CodecId::Gzip) => {
            let c: gzip::GzipCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| gzip::GzipCodec::default());
            Ok(AnyCodec::Gzip(c))
        }
        Some(CodecId::Blosc) => {
            let c: blosc::BloscCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| blosc::BloscCodec::default());
            Ok(AnyCodec::Blosc(c))
        }
        Some(CodecId::Zlib) => {
            let c: zlib::ZlibCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| zlib::ZlibCodec::default());
            Ok(AnyCodec::Zlib(c))
        }
        Some(CodecId::Zstd) => {
            let c: zstd::ZstdCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| zstd::ZstdCodec::default());
            Ok(AnyCodec::Zstd(c))
        }
        Some(CodecId::Lz4) => {
            let c: lz4::Lz4Codec = serde_json::from_value(config)
                .unwrap_or_else(|_| lz4::Lz4Codec::default());
            Ok(AnyCodec::Lz4(c))
        }
        Some(CodecId::Sharding) => {
            let c: sharding::ShardingCodec = serde_json::from_value(config)
                .unwrap_or_else(|_| sharding::ShardingCodec::default());
            Ok(AnyCodec::Sharding(c))
        }
        Some(CodecId::FixedScaleOffset) => {
            let c: fixedscaleoffset::FixedScaleOffsetCodec = serde_json::from_value(config)
                .map_err(|e| {
                    ZarrError::Codec(format!("Failed to parse FixedScaleOffsetCodec: {e}"))
                })?;
            Ok(AnyCodec::FixedScaleOffset(c))
        }
        None => Err(ZarrError::Codec(format!("Unknown codec: {}", env.name))),
    }
}

/// Parse a list of codecs from JSON values.
pub fn parse_codecs(values: &[serde_json::Value]) -> ZarrResult<Vec<AnyCodec>> {
    values.iter().map(parse_codec).collect()
}
