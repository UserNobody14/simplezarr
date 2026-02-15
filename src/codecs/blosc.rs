use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Blosc sub-compressor and shuffle types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BloscCname {
    Lz4,
    Lz4hc,
    Blosclz,
    Zstd,
    Snappy,
    Zlib,
}

impl std::fmt::Display for BloscCname {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BloscCname::Lz4 => write!(f, "lz4"),
            BloscCname::Lz4hc => write!(f, "lz4hc"),
            BloscCname::Blosclz => write!(f, "blosclz"),
            BloscCname::Zstd => write!(f, "zstd"),
            BloscCname::Snappy => write!(f, "snappy"),
            BloscCname::Zlib => write!(f, "zlib"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloscShuffle {
    NoShuffle,
    Shuffle,
    BitShuffle,
}

impl Serialize for BloscShuffle {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            BloscShuffle::NoShuffle => serializer.serialize_str("noshuffle"),
            BloscShuffle::Shuffle => serializer.serialize_str("shuffle"),
            BloscShuffle::BitShuffle => serializer.serialize_str("bitshuffle"),
        }
    }
}

impl<'de> Deserialize<'de> for BloscShuffle {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(deserializer)?;
        match &v {
            serde_json::Value::String(s) => match s.to_lowercase().as_str() {
                "noshuffle" | "0" => Ok(BloscShuffle::NoShuffle),
                "shuffle" | "1" => Ok(BloscShuffle::Shuffle),
                "bitshuffle" | "2" => Ok(BloscShuffle::BitShuffle),
                other => Err(serde::de::Error::custom(format!(
                    "Unknown blosc shuffle: {other}"
                ))),
            },
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(0) => Ok(BloscShuffle::NoShuffle),
                Some(1) => Ok(BloscShuffle::Shuffle),
                Some(2) => Ok(BloscShuffle::BitShuffle),
                _ => Err(serde::de::Error::custom(format!(
                    "Unknown blosc shuffle int: {n}"
                ))),
            },
            _ => Err(serde::de::Error::custom(
                "Expected string or int for blosc shuffle",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// BloscCodec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloscCodec {
    #[serde(default)]
    pub typesize: Option<usize>,
    #[serde(default = "default_cname")]
    pub cname: BloscCname,
    #[serde(default = "default_clevel")]
    pub clevel: i32,
    #[serde(default)]
    pub shuffle: Option<BloscShuffle>,
    #[serde(default)]
    pub blocksize: usize,
}

fn default_cname() -> BloscCname {
    BloscCname::Zstd
}

fn default_clevel() -> i32 {
    5
}

impl Default for BloscCodec {
    fn default() -> Self {
        Self {
            typesize: None,
            cname: BloscCname::Zstd,
            clevel: 5,
            shuffle: Some(BloscShuffle::NoShuffle),
            blocksize: 0,
        }
    }
}

impl BloscCodec {
    /// Decompress blosc-compressed data using the pure-Rust `blusc` library.
    /// Runs on a blocking thread since decompression can be CPU-intensive.
    pub async fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || blosc_decompress(&data))
            .await
            .map_err(|e| ZarrError::Decode(format!("Blosc task join error: {e}")))?
    }

    /// Compress data using the pure-Rust `blusc` library.
    pub async fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let data = data.to_vec();
        let clevel = self.clevel;
        let shuffle = self.shuffle.unwrap_or(BloscShuffle::NoShuffle);
        let typesize = self.typesize.unwrap_or(1);
        tokio::task::spawn_blocking(move || blosc_compress(&data, clevel, shuffle, typesize))
            .await
            .map_err(|e| ZarrError::Encode(format!("Blosc task join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust blosc wrappers via `blusc`
// ---------------------------------------------------------------------------

/// Read the uncompressed size from a blosc header.
/// The blosc1 header layout (16 bytes):
///   byte 0: blosc version
///   byte 1: blosc version format
///   byte 2: flags
///   byte 3: typesize
///   bytes 4..8: nbytes (uncompressed, little-endian u32)
///   bytes 8..12: blocksize
///   bytes 12..16: cbytes (compressed, little-endian u32)
fn blosc_header_nbytes(data: &[u8]) -> ZarrResult<usize> {
    if data.len() < 16 {
        return Err(ZarrError::Decode(
            "Blosc buffer too small for header (need >= 16 bytes)".into(),
        ));
    }
    let nbytes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    Ok(nbytes)
}

fn blosc_decompress(data: &[u8]) -> ZarrResult<Vec<u8>> {
    let nbytes = blosc_header_nbytes(data)?;
    let mut output = vec![0u8; nbytes];
    let result = blusc::blosc2_decompress(data, &mut output);
    if result < 0 {
        return Err(ZarrError::Decode(format!(
            "Blosc decompress returned error code: {result}"
        )));
    }
    Ok(output)
}

fn blosc_compress(
    data: &[u8],
    clevel: i32,
    shuffle: BloscShuffle,
    typesize: usize,
) -> ZarrResult<Vec<u8>> {
    let shuffle_int = match shuffle {
        BloscShuffle::NoShuffle => blusc::BLOSC_NOSHUFFLE as i32,
        BloscShuffle::Shuffle => blusc::BLOSC_SHUFFLE as i32,
        BloscShuffle::BitShuffle => blusc::BLOSC_BITSHUFFLE as i32,
    };

    let mut compressed = vec![0u8; data.len() + blusc::BLOSC2_MAX_OVERHEAD];
    let cbytes =
        blusc::blosc2_compress(clevel, shuffle_int, typesize, data, &mut compressed);
    if cbytes < 0 {
        return Err(ZarrError::Encode(format!(
            "Blosc compress returned error code: {cbytes}"
        )));
    }
    compressed.truncate(cbytes as usize);
    Ok(compressed)
}
