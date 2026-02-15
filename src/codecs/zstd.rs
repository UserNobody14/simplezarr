use crate::error::{ZarrError, ZarrResult};
use serde::{Deserialize, Serialize};

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
        zstd::bulk::decompress(data, 0) // 0 = auto-detect size from frame header
            .map_err(|e| ZarrError::Decode(format!("Zstd decompress failed: {e}")))
    }

    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        zstd::bulk::compress(data, self.level)
            .map_err(|e| ZarrError::Encode(format!("Zstd compress failed: {e}")))
    }
}
