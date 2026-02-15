use crate::error::{ZarrError, ZarrResult};
use async_trait::async_trait;
use bytes::Bytes;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// StorageBackend trait
// ---------------------------------------------------------------------------

/// Async storage abstraction, modelled after the Haskell `StorageBackend`.
///
/// Implementations can target local filesystem, S3, GCS, Azure, or in-memory
/// stores.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Fetch the contents at `path`.
    /// Returns `Ok(None)` when the key does not exist (rather than an error).
    async fn get(&self, path: &str) -> ZarrResult<Option<Bytes>>;

    /// List immediate children under `prefix`.
    async fn list(&self, prefix: &str) -> ZarrResult<Vec<String>>;

    /// Join a base path with a relative segment.
    fn join(&self, base: &str, segment: &str) -> String;
}

// ---------------------------------------------------------------------------
// LocalBackend  (tokio::fs)
// ---------------------------------------------------------------------------

/// Simple local-filesystem backend using `tokio::fs`.
#[derive(Debug, Clone)]
pub struct LocalBackend {
    root: PathBuf,
}

impl LocalBackend {
    /// Create a new backend rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

#[async_trait]
impl StorageBackend for LocalBackend {
    async fn get(&self, path: &str) -> ZarrResult<Option<Bytes>> {
        let full = self.resolve(path);
        match tokio::fs::read(&full).await {
            Ok(data) => {
                if data.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(Bytes::from(data)))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ZarrError::Storage(format!(
                "Failed to read {}: {e}",
                full.display()
            ))),
        }
    }

    async fn list(&self, prefix: &str) -> ZarrResult<Vec<String>> {
        let dir = self.resolve(prefix);
        let mut entries = Vec::new();
        let mut reader = tokio::fs::read_dir(&dir).await.map_err(|e| {
            ZarrError::Storage(format!("Failed to list {}: {e}", dir.display()))
        })?;
        while let Some(entry) = reader.next_entry().await.map_err(|e| {
            ZarrError::Storage(format!("Failed to read entry in {}: {e}", dir.display()))
        })? {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
        Ok(entries)
    }

    fn join(&self, base: &str, segment: &str) -> String {
        let p = Path::new(base).join(segment);
        p.to_string_lossy().into_owned()
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreBackend  (wraps object_store crate)
// ---------------------------------------------------------------------------

/// Backend that wraps any [`object_store::ObjectStore`] implementation.
pub struct ObjectStoreBackend {
    store: Box<dyn object_store::ObjectStore>,
    prefix: String,
}

impl ObjectStoreBackend {
    pub fn new(store: Box<dyn object_store::ObjectStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            prefix: prefix.into(),
        }
    }

    fn full_path(&self, path: &str) -> object_store::path::Path {
        if self.prefix.is_empty() {
            object_store::path::Path::from(path)
        } else {
            object_store::path::Path::from(format!("{}/{}", self.prefix, path))
        }
    }
}

#[async_trait]
impl StorageBackend for ObjectStoreBackend {
    async fn get(&self, path: &str) -> ZarrResult<Option<Bytes>> {
        let location = self.full_path(path);
        match self.store.get(&location).await {
            Ok(result) => {
                let data = result.bytes().await.map_err(|e| {
                    ZarrError::Storage(format!("Failed to read bytes from {path}: {e}"))
                })?;
                if data.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(data))
                }
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(ZarrError::Storage(format!(
                "Object store error for {path}: {e}"
            ))),
        }
    }

    async fn list(&self, prefix: &str) -> ZarrResult<Vec<String>> {
        use futures::TryStreamExt;
        let location = self.full_path(prefix);
        let mut entries = Vec::new();
        let mut stream = self.store.list(Some(&location));
        while let Some(meta) = stream.try_next().await.map_err(|e| {
            ZarrError::Storage(format!("Object store list error for {prefix}: {e}"))
        })? {
            entries.push(meta.location.to_string());
        }
        Ok(entries)
    }

    fn join(&self, base: &str, segment: &str) -> String {
        if base.is_empty() {
            segment.to_string()
        } else {
            format!("{base}/{segment}")
        }
    }
}
