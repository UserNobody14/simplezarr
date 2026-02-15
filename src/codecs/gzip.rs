use crate::error::{ZarrError, ZarrResult};
use flate2::read::{GzDecoder, GzEncoder};
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GzipCodec {
    #[serde(default = "default_level")]
    pub level: u32,
}

fn default_level() -> u32 {
    5
}

impl Default for GzipCodec {
    fn default() -> Self {
        Self { level: 5 }
    }
}

impl GzipCodec {
    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| ZarrError::Decode(format!("Gzip decompress failed: {e}")))?;
        Ok(out)
    }

    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let level = Compression::new(self.level.min(9));
        let mut encoder = GzEncoder::new(data, level);
        let mut out = Vec::new();
        encoder
            .read_to_end(&mut out)
            .map_err(|e| ZarrError::Encode(format!("Gzip compress failed: {e}")))?;
        Ok(out)
    }
}
