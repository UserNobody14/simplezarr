use thiserror::Error;

pub type ZarrResult<T> = Result<T, ZarrError>;

#[derive(Error, Debug)]
pub enum ZarrError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Metadata error: {0}")]
    Metadata(String),

    #[error("Decode error: {0}")]
    Decode(String),

    #[error("Encode error: {0}")]
    Encode(String),

    #[error("Type conversion error: {0}")]
    TypeConversion(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Other(String),
}
