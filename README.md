# simplezarr

A lightweight, async Rust crate for reading [Zarr](https://zarr.dev/) V2 arrays and groups.

## Features

- **Zarr V2 support** -- read arrays and groups stored in the Zarr V2 format
- **Async / Tokio** -- all I/O is fully async; chunks are fetched concurrently
- **Pluggable storage backends** -- ships with `LocalBackend` (local filesystem via `tokio::fs`) and `ObjectStoreBackend` (wraps any [`object_store`](https://docs.rs/object_store) implementation for S3, GCS, Azure, etc.)
- **Consolidated metadata** -- transparently reads `.zmetadata` when available, with fallback to per-array `.zarray` files
- **Rich type system** -- preserves the full Zarr type hierarchy (bool, int8–int64, uint8–uint64, float16/32/64, complex64/128, string, bytes) without forcing lossy f64 conversion
- **Compression codecs** -- built-in support for Blosc, Gzip, Zlib, Zstd, and LZ4
- **Both C and Fortran array order**

## Quick start

Add the dependency:

```toml
[dependencies]
simplezarr = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Read a single array

```rust
use std::sync::Arc;
use simplezarr::store::LocalBackend;
use simplezarr::v2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(LocalBackend::new("path/to/dataset.zarr"));

    // Open and inspect metadata
    let array = v2::open(store.clone(), "temperature").await?;
    println!("shape: {:?}", array.metadata.shape);
    println!("dtype: {:?}", array.metadata.data_type);

    // Fetch a single chunk by its indices
    let chunk = array.get_chunk(&[0, 0]).await?;

    Ok(())
}
```

### Read a group of arrays

```rust
use std::sync::Arc;
use simplezarr::store::LocalBackend;
use simplezarr::v2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(LocalBackend::new("path/to/dataset.zarr"));

    // Open a group (uses consolidated metadata if available)
    let group = v2::open_group(
        store,
        ".",
        &["temperature", "precipitation"],
    ).await?;

    // Access individual arrays
    if let Some(temp) = group.get_array("temperature") {
        let first_key: Vec<usize> = vec![0; temp.metadata.shape.len()];
        let chunk = temp.get_chunk(&first_key).await?;
        let data = chunk.to_f64_vec()?;
        println!("temperature: {} elements", data.len());
    }

    Ok(())
}
```

### Cloud storage via `object_store`

```rust
use std::sync::Arc;
use simplezarr::store::ObjectStoreBackend;
use simplezarr::v2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s3 = object_store::aws::AmazonS3Builder::from_env()
        .with_bucket_name("my-bucket")
        .build()?;

    let store = Arc::new(ObjectStoreBackend::new(Box::new(s3), "datasets/weather"));

    let array = v2::open(store, "temperature").await?;
    let first_key: Vec<usize> = vec![0; array.metadata.shape.len()];
    let chunk = array.get_chunk(&first_key).await?;
    let data = chunk.to_f64_vec()?;

    Ok(())
}
```

## Supported data types

| Zarr dtype | Rust representation |
|---|---|
| `bool` | `bool` |
| `int8` – `int64` | `i8` – `i64` |
| `uint8` – `uint64` | `u8` – `u64` |
| `float16` | `half::f16` |
| `float32` / `float64` | `f32` / `f64` |
| `complex64` / `complex128` | `num_complex::Complex<f32>` / `Complex<f64>` |
| string / bytes | `String` / `Vec<u8>` |

## Supported compressors

| Compressor | V2 `compressor.id` |
|---|---|
| Blosc (lz4, lz4hc, blosclz, zstd, snappy, zlib) | `blosc` |
| Gzip | `gzip` |
| Zlib | `zlib` |
| Zstd | `zstd` |
| LZ4 | `lz4` |

Both little-endian and big-endian byte orders are supported.

## License

MIT
