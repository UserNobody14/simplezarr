pub mod array;
pub mod codecs;
pub mod error;
pub mod group;
pub mod metadata;
pub mod store;
pub mod types;
pub mod v2;

// Re-export key types at crate root for convenience.
pub use array::{UnifiedMetadata, UnifiedZarrArray};
pub use error::{ZarrError, ZarrResult};
pub use group::{UnifiedGroupMetadata, UnifiedZarrGroup};
pub use store::{LocalBackend, ObjectStoreBackend, StorageBackend};
pub use types::{
    ArrayOrder, DataType, Endian, FillValue, ZarrValue, ZarrVectorValue,
};
