use serde::{Deserialize, Serialize};

/// Sharding codec (placeholder).
///
/// Full shard decoding requires additional context (inner chunk shape,
/// index codec, inner codecs) that goes beyond simple byte-in / byte-out.
/// This struct captures the configuration so higher-level code can
/// detect and handle sharded arrays.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardingCodec {
    #[serde(default)]
    pub chunk_shape: Vec<usize>,
    #[serde(default)]
    pub codecs: Vec<serde_json::Value>,
}
