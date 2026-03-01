use crate::error::{ZarrError, ZarrResult};
use crate::types::{DataType, Endian, ZarrValue};
use half::f16;
use num_complex::Complex;
use zerocopy::Ref;
use zerocopy::byteorder::little_endian;

// ---------------------------------------------------------------------------
// ZarrVec  (typed chunk data)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ZarrVec {
    VBool(Vec<bool>),
    VInt8(Vec<i8>),
    VInt16(Vec<i16>),
    VInt32(Vec<i32>),
    VInt64(Vec<i64>),
    VUInt8(Vec<u8>),
    VUInt16(Vec<u16>),
    VUInt32(Vec<u32>),
    VUInt64(Vec<u64>),
    VFloat16(Vec<f16>),
    VFloat32(Vec<f32>),
    VFloat64(Vec<f64>),
    VComplex64(Vec<Complex<f32>>),
    VComplex128(Vec<Complex<f64>>),
    VString(Vec<String>),
    VBytes(Vec<Vec<u8>>),
    VWithNulls(DataType, Vec<Option<ZarrValue>>),
}

impl ZarrVec {
    /// Number of elements in the vector.
    pub fn len(&self) -> usize {
        match self {
            ZarrVec::VBool(v) => v.len(),
            ZarrVec::VInt8(v) => v.len(),
            ZarrVec::VInt16(v) => v.len(),
            ZarrVec::VInt32(v) => v.len(),
            ZarrVec::VInt64(v) => v.len(),
            ZarrVec::VUInt8(v) => v.len(),
            ZarrVec::VUInt16(v) => v.len(),
            ZarrVec::VUInt32(v) => v.len(),
            ZarrVec::VUInt64(v) => v.len(),
            ZarrVec::VFloat16(v) => v.len(),
            ZarrVec::VFloat32(v) => v.len(),
            ZarrVec::VFloat64(v) => v.len(),
            ZarrVec::VComplex64(v) => v.len(),
            ZarrVec::VComplex128(v) => v.len(),
            ZarrVec::VString(v) => v.len(),
            ZarrVec::VBytes(v) => v.len(),
            ZarrVec::VWithNulls(_, v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lossily convert the entire vector to `Vec<f64>`.
    pub fn to_f64_vec(&self) -> ZarrResult<Vec<f64>> {
        match self {
            ZarrVec::VBool(v) => Ok(v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()),
            ZarrVec::VInt8(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VInt16(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VInt32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VInt64(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VUInt8(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VUInt16(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VUInt32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VUInt64(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VFloat16(v) => Ok(v.iter().map(|x| x.to_f64()).collect()),
            ZarrVec::VFloat32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVec::VFloat64(v) => Ok(v.clone()),
            ZarrVec::VComplex64(v) => Ok(v.iter().map(|c| c.re as f64).collect()),
            ZarrVec::VComplex128(v) => Ok(v.iter().map(|c| c.re).collect()),
            ZarrVec::VString(_) => Err(ZarrError::TypeConversion(
                "Cannot convert String to f64".into(),
            )),
            ZarrVec::VBytes(_) => Err(ZarrError::TypeConversion(
                "Cannot convert Bytes to f64".into(),
            )),
            ZarrVec::VWithNulls(_, v) => Ok(v
                .iter()
                .map(|opt| opt.as_ref().and_then(|zv| zv.to_f64()).unwrap_or(f64::NAN))
                .collect()),
        }
    }

    /// Convert to `Vec<Option<ZarrValue>>`, wrapping each element.
    pub fn to_maybe_values(&self) -> Vec<Option<ZarrValue>> {
        match self {
            ZarrVec::VBool(v) => v.iter().map(|x| Some(ZarrValue::Bool(*x))).collect(),
            ZarrVec::VInt8(v) => v.iter().map(|x| Some(ZarrValue::Int8(*x))).collect(),
            ZarrVec::VInt16(v) => v.iter().map(|x| Some(ZarrValue::Int16(*x))).collect(),
            ZarrVec::VInt32(v) => v.iter().map(|x| Some(ZarrValue::Int32(*x))).collect(),
            ZarrVec::VInt64(v) => v.iter().map(|x| Some(ZarrValue::Int64(*x))).collect(),
            ZarrVec::VUInt8(v) => v.iter().map(|x| Some(ZarrValue::UInt8(*x))).collect(),
            ZarrVec::VUInt16(v) => v.iter().map(|x| Some(ZarrValue::UInt16(*x))).collect(),
            ZarrVec::VUInt32(v) => v.iter().map(|x| Some(ZarrValue::UInt32(*x))).collect(),
            ZarrVec::VUInt64(v) => v.iter().map(|x| Some(ZarrValue::UInt64(*x))).collect(),
            ZarrVec::VFloat16(v) => v.iter().map(|x| Some(ZarrValue::Float16(*x))).collect(),
            ZarrVec::VFloat32(v) => v.iter().map(|x| Some(ZarrValue::Float32(*x))).collect(),
            ZarrVec::VFloat64(v) => v.iter().map(|x| Some(ZarrValue::Float64(*x))).collect(),
            ZarrVec::VComplex64(v) => v.iter().map(|x| Some(ZarrValue::Complex64(*x))).collect(),
            ZarrVec::VComplex128(v) => v.iter().map(|x| Some(ZarrValue::Complex128(*x))).collect(),
            ZarrVec::VString(v) => v
                .iter()
                .map(|x| Some(ZarrValue::String(x.clone())))
                .collect(),
            ZarrVec::VBytes(v) => v
                .iter()
                .map(|x| Some(ZarrValue::Bytes(x.clone())))
                .collect(),
            ZarrVec::VWithNulls(_, v) => v.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw bytes -> typed vector
// ---------------------------------------------------------------------------

/// Transmutes `Vec<u8>` into `Vec<i16>` in-place (zero allocation).
/// Caller must ensure `data.len() % 2 == 0` and bytes are already in native endianness.
#[inline]
fn vec_u8_into_vec_i16(data: Vec<u8>) -> ZarrResult<Vec<i16>> {
    if data.len() % 2 != 0 {
        return Err(ZarrError::Decode(format!(
            "Data length {} is not a multiple of 2",
            data.len()
        )));
    }
    let len = data.len() / 2;
    let capacity = data.capacity() / 2;
    let mut data = std::mem::ManuallyDrop::new(data);
    let ptr = data.as_mut_ptr() as *mut i16;
    Ok(unsafe { Vec::from_raw_parts(ptr, len, capacity) })
}

/// Interpret raw bytes as a typed vector according to `endian` and `dtype`.
/// Uses zerocopy for validation and zero-allocation conversion when endianness
/// matches the target, or in-place byte swap when it does not.
pub fn bytes_to_zarr_vector(
    endian: Endian,
    dtype: DataType,
    mut data: Vec<u8>,
) -> ZarrResult<ZarrVec> {
    match dtype {
        DataType::Int16 => {
            // Validate layout with zerocopy (both endianness types have same layout: 2 bytes/elem)
            Ref::<_, [little_endian::I16]>::from_bytes(data.as_slice()).map_err(|_| {
                ZarrError::Decode(format!(
                    "Invalid byte layout for i16: length {}",
                    data.len()
                ))
            })?;

            let vsv: Vec<i16> = match endian {
                Endian::Little | Endian::NotApplicable => {
                    #[cfg(target_endian = "little")]
                    {
                        // Same endian: zero-copy reinterpret
                        vec_u8_into_vec_i16(data)?
                    }
                    #[cfg(target_endian = "big")]
                    {
                        // Need byte-swap; do in-place then transmute (no extra alloc)
                        data.chunks_exact_mut(2).for_each(|c| c.swap(0, 1));
                        vec_u8_into_vec_i16(data)?
                    }
                }
                Endian::Big => {
                    #[cfg(target_endian = "big")]
                    {
                        vec_u8_into_vec_i16(data)?
                    }
                    #[cfg(target_endian = "little")]
                    {
                        data.chunks_exact_mut(2).for_each(|c| c.swap(0, 1));
                        vec_u8_into_vec_i16(data)?
                    }
                }
            };
            Ok(ZarrVec::VInt16(vsv))
        }
        _ => Err(ZarrError::Decode(format!(
            "Unsupported data type: {dtype:?}"
        ))),
    }
}

/// Create a filled chunk vector by replicating a scalar value.
pub fn fill_chunk(value: &ZarrValue, chunk_shape: &[usize]) -> ZarrVec {
    let total: usize = chunk_shape.iter().product();
    match value {
        ZarrValue::Bool(b) => ZarrVec::VBool(vec![*b; total]),
        ZarrValue::Int8(v) => ZarrVec::VInt8(vec![*v; total]),
        ZarrValue::Int16(v) => ZarrVec::VInt16(vec![*v; total]),
        ZarrValue::Int32(v) => ZarrVec::VInt32(vec![*v; total]),
        ZarrValue::Int64(v) => ZarrVec::VInt64(vec![*v; total]),
        ZarrValue::UInt8(v) => ZarrVec::VUInt8(vec![*v; total]),
        ZarrValue::UInt16(v) => ZarrVec::VUInt16(vec![*v; total]),
        ZarrValue::UInt32(v) => ZarrVec::VUInt32(vec![*v; total]),
        ZarrValue::UInt64(v) => ZarrVec::VUInt64(vec![*v; total]),
        ZarrValue::Float16(v) => ZarrVec::VFloat16(vec![*v; total]),
        ZarrValue::Float32(v) => ZarrVec::VFloat32(vec![*v; total]),
        ZarrValue::Float64(v) => ZarrVec::VFloat64(vec![*v; total]),
        ZarrValue::Complex64(v) => ZarrVec::VComplex64(vec![*v; total]),
        ZarrValue::Complex128(v) => ZarrVec::VComplex128(vec![*v; total]),
        ZarrValue::String(s) => ZarrVec::VString(vec![s.clone(); total]),
        ZarrValue::Bytes(b) => ZarrVec::VBytes(vec![b.clone(); total]),
        ZarrValue::Null(dt) => ZarrVec::VWithNulls(*dt, vec![None; total]),
    }
}
