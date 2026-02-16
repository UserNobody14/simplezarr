use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};
use std::ffi::CStr;

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
    /// Decompress blosc-compressed data.
    /// Runs on a blocking thread since decompression can be CPU-intensive.
    pub async fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let data = data.to_vec();
        tokio::task::spawn_blocking(move || blosc_decompress(&data))
            .await
            .map_err(|e| ZarrError::Decode(format!("Blosc task join error: {e}")))?
    }

    /// Compress data using blosc.
    pub async fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let data = data.to_vec();
        let clevel = self.clevel;
        let shuffle = self.shuffle.unwrap_or(BloscShuffle::NoShuffle);
        let typesize = self.typesize.unwrap_or(1);
        let cname = self.cname;
        let blocksize = self.blocksize;
        tokio::task::spawn_blocking(move || {
            blosc_compress(&data, clevel, shuffle, typesize, cname, blocksize)
        })
        .await
        .map_err(|e| ZarrError::Encode(format!("Blosc task join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Blosc FFI wrappers
// ---------------------------------------------------------------------------

/// Map a `BloscCname` to the corresponding C string expected by blosc.
fn compressor_as_cstr(cname: BloscCname) -> &'static CStr {
    match cname {
        BloscCname::Lz4 => c"lz4",
        BloscCname::Lz4hc => c"lz4hc",
        BloscCname::Blosclz => c"blosclz",
        BloscCname::Zstd => c"zstd",
        BloscCname::Snappy => c"snappy",
        BloscCname::Zlib => c"zlib",
    }
}

/// Validate a blosc compressed buffer and return the uncompressed size.
/// Returns `None` if the buffer is invalid.
fn blosc_validate(data: &[u8]) -> Option<usize> {
    let mut nbytes: usize = 0;
    let result =
        unsafe { blosc_src::blosc_cbuffer_validate(data.as_ptr().cast(), data.len(), &mut nbytes) };
    if result == 0 { Some(nbytes) } else { None }
}

/// Decompress a blosc-compressed buffer.
///
/// Uses `blosc_decompress_ctx` which is thread-safe and does not require
/// `blosc_init()`.
fn blosc_decompress(data: &[u8]) -> ZarrResult<Vec<u8>> {
    let nbytes = blosc_validate(data)
        .ok_or_else(|| ZarrError::Decode("Blosc encoded value is invalid".into()))?;

    if nbytes == 0 {
        return Ok(Vec::new());
    }

    let mut output = vec![0u8; nbytes];
    let result = unsafe {
        blosc_src::blosc_decompress_ctx(
            data.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            output.len(),
            1, // numinternalthreads
        )
    };
    if result < 0 {
        return Err(ZarrError::Decode(format!(
            "Blosc decompress returned error code: {result}"
        )));
    }
    Ok(output)
}

/// Compress data using blosc.
///
/// Uses `blosc_compress_ctx` which is thread-safe, does not require
/// `blosc_init()`, and accepts the compressor name directly (no global state).
fn blosc_compress(
    data: &[u8],
    clevel: i32,
    shuffle: BloscShuffle,
    typesize: usize,
    cname: BloscCname,
    blocksize: usize,
) -> ZarrResult<Vec<u8>> {
    let shuffle_int = match shuffle {
        BloscShuffle::NoShuffle => blosc_src::BLOSC_NOSHUFFLE as i32,
        BloscShuffle::Shuffle => blosc_src::BLOSC_SHUFFLE as i32,
        BloscShuffle::BitShuffle => blosc_src::BLOSC_BITSHUFFLE as i32,
    };

    let destsize = data.len() + blosc_src::BLOSC_MAX_OVERHEAD as usize;
    let mut compressed = vec![0u8; destsize];

    let cbytes = unsafe {
        blosc_src::blosc_compress_ctx(
            clevel,
            shuffle_int,
            typesize,
            data.len(),
            data.as_ptr().cast(),
            compressed.as_mut_ptr().cast(),
            destsize,
            compressor_as_cstr(cname).as_ptr(),
            blocksize,
            1, // numinternalthreads
        )
    };

    if cbytes < 0 {
        return Err(ZarrError::Encode(format!(
            "Blosc compress returned error code: {cbytes}"
        )));
    }
    compressed.truncate(cbytes as usize);
    Ok(compressed)
}
