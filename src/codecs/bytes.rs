use crate::error::ZarrResult;
use crate::types::Endian;
use serde::{Deserialize, Serialize};

/// Bytes codec: pass-through that records endianness metadata.
/// Actual endian conversion happens at the typed-interpretation layer,
/// not during raw byte decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytesCodec {
    pub endian: Option<Endian>,
}

impl Default for BytesCodec {
    fn default() -> Self {
        Self {
            endian: Some(Endian::Little),
        }
    }
}

impl BytesCodec {
    pub fn new(endian: Endian) -> Self {
        Self {
            endian: Some(endian),
        }
    }

    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        Ok(data.to_vec())
    }

    pub fn encode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        Ok(data.to_vec())
    }
}

// Custom serde for Endian so it works in JSON configs
impl Serialize for Endian {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Endian::Little => serializer.serialize_str("little"),
            Endian::Big => serializer.serialize_str("big"),
            Endian::NotApplicable => serializer.serialize_str("not_applicable"),
        }
    }
}

impl<'de> Deserialize<'de> for Endian {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "little" => Ok(Endian::Little),
            "big" => Ok(Endian::Big),
            "not_applicable" | "na" | "" => Ok(Endian::NotApplicable),
            other => Err(serde::de::Error::custom(format!(
                "Unknown endian: {other}"
            ))),
        }
    }
}
