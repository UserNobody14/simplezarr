use std::collections::HashMap;

use crate::array::{load_array, UnifiedZarrArray};
use crate::error::{ZarrError, ZarrResult};

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

    /// Load every array in the group concurrently, returning name -> Vec<f64>.
    pub async fn load_all(&self) -> ZarrResult<HashMap<String, Vec<f64>>> {
        let handles: Vec<_> = self
            .arrays
            .iter()
            .map(|(name, array)| {
                let name = name.clone();
                // We need a &UnifiedZarrArray, but we're iterating the map.
                // Since load_array takes &UnifiedZarrArray and we can't move
                // out of the map, we'll collect the futures inline.
                let getter = array.get_chunk.clone();
                let md = array.metadata.clone();
                tokio::spawn(async move {
                    let array_ref = UnifiedZarrArray {
                        metadata: md,
                        get_chunk: getter,
                    };
                    let data = load_array(&array_ref).await?;
                    Ok::<_, ZarrError>((name, data))
                })
            })
            .collect();

        let mut results = HashMap::new();
        let mut errors = Vec::new();

        for handle in handles {
            match handle.await {
                Ok(Ok((name, data))) => {
                    results.insert(name, data);
                }
                Ok(Err(e)) => errors.push(e),
                Err(e) => errors.push(ZarrError::Other(format!("Task join error: {e}"))),
            }
        }

        if let Some(err) = errors.into_iter().next() {
            return Err(err);
        }

        Ok(results)
    }
}
