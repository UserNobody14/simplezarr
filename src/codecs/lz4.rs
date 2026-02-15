use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};

const LZ4_SIZE_PREFIX_BYTES: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lz4Codec {
    #[serde(default = "default_acceleration")]
    pub acceleration: i32,
}

fn default_acceleration() -> i32 {
    1
}

impl Default for Lz4Codec {
    fn default() -> Self {
        Self { acceleration: 1 }
    }
}

impl Lz4Codec {
    /// Decode an LZ4 block that has a 4-byte little-endian size prefix
    /// (matching the Zarr / numcodecs convention).
    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        if data.len() < LZ4_SIZE_PREFIX_BYTES {
            return Err(ZarrError::Decode(
                "LZ4 decode: compressed buffer missing 4-byte size prefix".into(),
            ));
        }

        let (prefix, payload) = data.split_at(LZ4_SIZE_PREFIX_BYTES);
        let dest_size =
            u32::from_le_bytes(prefix.try_into().unwrap()) as usize;

        let decompressed = lz4_flex::block::decompress(payload, dest_size)
            .map_err(|e| ZarrError::Decode(format!("LZ4 decompress failed: {e}")))?;

        if decompressed.len() != dest_size {
            return Err(ZarrError::Decode(format!(
                "LZ4 decompression error: expected {} bytes, got {}",
                dest_size,
                decompressed.len()
            )));
        }

        Ok(decompressed)
    }

    /// Encode an LZ4 block, prepending a 4-byte little-endian size prefix.
    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let orig_size = data.len() as u32;
        let compressed = lz4_flex::block::compress(data);
        let mut out = Vec::with_capacity(LZ4_SIZE_PREFIX_BYTES + compressed.len());
        out.extend_from_slice(&orig_size.to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok(out)
    }
}
