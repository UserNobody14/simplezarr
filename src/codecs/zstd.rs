use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZstdCodec {
    #[serde(default = "default_level")]
    pub level: i32,
}

fn default_level() -> i32 {
    5
}

impl Default for ZstdCodec {
    fn default() -> Self {
        Self { level: 5 }
    }
}

impl ZstdCodec {
    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        // Use streaming decoder -- handles frames that lack a content-size field
        // (common with numcodecs' zstd output).
        let mut decoder = zstd::Decoder::new(data)
            .map_err(|e| ZarrError::Decode(format!("Zstd decoder init failed: {e}")))?;
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| ZarrError::Decode(format!("Zstd decompress failed: {e}")))?;
        Ok(out)
    }

    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        zstd::bulk::compress(data, self.level)
            .map_err(|e| ZarrError::Encode(format!("Zstd compress failed: {e}")))
    }
}
