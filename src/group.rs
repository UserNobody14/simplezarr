use std::collections::HashMap;

use crate::array::UnifiedZarrArray;

// ---------------------------------------------------------------------------
// UnifiedGroupMetadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UnifiedGroupMetadata {
    pub zarr_format: u32,
    pub attributes: Option<serde_json::Map<String, serde_json::Value>>,
    pub consolidated: bool,
    pub array_names: Vec<String>,
    pub path: String,
}

// ---------------------------------------------------------------------------
// UnifiedZarrGroup
// ---------------------------------------------------------------------------

pub struct UnifiedZarrGroup {
    pub metadata: UnifiedGroupMetadata,
    pub arrays: HashMap<String, UnifiedZarrArray>,
}

impl std::fmt::Debug for UnifiedZarrGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedZarrGroup")
            .field("metadata", &self.metadata)
            .field("arrays", &self.arrays.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl UnifiedZarrGroup {
    // Convenience accessors (mirrors Haskell helpers)

    pub fn zarr_format(&self) -> u32 {
        self.metadata.zarr_format
    }

    pub fn attributes(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.metadata.attributes.as_ref()
    }

    pub fn is_consolidated(&self) -> bool {
        self.metadata.consolidated
    }

    pub fn array_names(&self) -> &[String] {
        &self.metadata.array_names
    }

    pub fn path(&self) -> &str {
        &self.metadata.path
    }

    pub fn get_array(&self, name: &str) -> Option<&UnifiedZarrArray> {
        self.arrays.get(name)
    }
}
