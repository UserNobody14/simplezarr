use crate::error::{ZarrError, ZarrResult};
use flate2::read::{ZlibDecoder, ZlibEncoder};
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZlibCodec {
    #[serde(default = "default_level")]
    pub level: u32,
}

fn default_level() -> u32 {
    1
}

impl Default for ZlibCodec {
    fn default() -> Self {
        Self { level: 1 }
    }
}

impl ZlibCodec {
    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(data);
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| ZarrError::Decode(format!("Zlib decompress failed: {e}")))?;
        Ok(out)
    }

    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        let level = Compression::new(self.level.min(9));
        let mut encoder = ZlibEncoder::new(data, level);
        let mut out = Vec::new();
        encoder
            .read_to_end(&mut out)
            .map_err(|e| ZarrError::Encode(format!("Zlib compress failed: {e}")))?;
        Ok(out)
    }
}
