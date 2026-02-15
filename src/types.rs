use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use half::f16;
use num_complex::Complex;
use std::io::Cursor;

use crate::error::{ZarrError, ZarrResult};

// ---------------------------------------------------------------------------
// Endian
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    Little,
    Big,
    NotApplicable,
}

// ---------------------------------------------------------------------------
// ArrayOrder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ArrayOrder {
    #[default]
    C,
    F,
}

// ---------------------------------------------------------------------------
// DataType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    String,
    Bytes,
}

impl DataType {
    /// Number of bytes per element for fixed-size types.
    pub fn byte_size(&self) -> Option<usize> {
        match self {
            DataType::Bool => Some(1),
            DataType::Int8 => Some(1),
            DataType::Int16 => Some(2),
            DataType::Int32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::UInt8 => Some(1),
            DataType::UInt16 => Some(2),
            DataType::UInt32 => Some(4),
            DataType::UInt64 => Some(8),
            DataType::Float16 => Some(2),
            DataType::Float32 => Some(4),
            DataType::Float64 => Some(8),
            DataType::Complex64 => Some(8),
            DataType::Complex128 => Some(16),
            DataType::String | DataType::Bytes => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ZarrValue  (scalar)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ZarrValue {
    Bool(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float16(f16),
    Float32(f32),
    Float64(f64),
    Complex64(Complex<f32>),
    Complex128(Complex<f64>),
    String(String),
    Bytes(Vec<u8>),
    Null(DataType),
}

impl ZarrValue {
    /// Return the [`DataType`] that this value belongs to.
    pub fn data_type(&self) -> DataType {
        match self {
            ZarrValue::Bool(_) => DataType::Bool,
            ZarrValue::Int8(_) => DataType::Int8,
            ZarrValue::Int16(_) => DataType::Int16,
            ZarrValue::Int32(_) => DataType::Int32,
            ZarrValue::Int64(_) => DataType::Int64,
            ZarrValue::UInt8(_) => DataType::UInt8,
            ZarrValue::UInt16(_) => DataType::UInt16,
            ZarrValue::UInt32(_) => DataType::UInt32,
            ZarrValue::UInt64(_) => DataType::UInt64,
            ZarrValue::Float16(_) => DataType::Float16,
            ZarrValue::Float32(_) => DataType::Float32,
            ZarrValue::Float64(_) => DataType::Float64,
            ZarrValue::Complex64(_) => DataType::Complex64,
            ZarrValue::Complex128(_) => DataType::Complex128,
            ZarrValue::String(_) => DataType::String,
            ZarrValue::Bytes(_) => DataType::Bytes,
            ZarrValue::Null(dt) => *dt,
        }
    }

    /// Lossily convert this scalar to `f64`.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            ZarrValue::Bool(true) => Some(1.0),
            ZarrValue::Bool(false) => Some(0.0),
            ZarrValue::Int8(v) => Some(*v as f64),
            ZarrValue::Int16(v) => Some(*v as f64),
            ZarrValue::Int32(v) => Some(*v as f64),
            ZarrValue::Int64(v) => Some(*v as f64),
            ZarrValue::UInt8(v) => Some(*v as f64),
            ZarrValue::UInt16(v) => Some(*v as f64),
            ZarrValue::UInt32(v) => Some(*v as f64),
            ZarrValue::UInt64(v) => Some(*v as f64),
            ZarrValue::Float16(v) => Some(v.to_f64()),
            ZarrValue::Float32(v) => Some(*v as f64),
            ZarrValue::Float64(v) => Some(*v),
            ZarrValue::Complex64(c) => Some(c.re as f64),
            ZarrValue::Complex128(c) => Some(c.re),
            ZarrValue::String(_) | ZarrValue::Bytes(_) | ZarrValue::Null(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// FillValue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FillValue {
    Value(ZarrValue),
    NaN,
    Infinity,
    NegativeInfinity,
}

impl FillValue {
    /// Convert to `f64`, with NaN / Infinity mapped accordingly.
    pub fn to_f64(&self) -> f64 {
        match self {
            FillValue::Value(v) => v.to_f64().unwrap_or(0.0),
            FillValue::NaN => f64::NAN,
            FillValue::Infinity => f64::INFINITY,
            FillValue::NegativeInfinity => f64::NEG_INFINITY,
        }
    }

    /// Return a concrete [`ZarrValue`] for the given dtype (used when filling
    /// chunks that are absent from storage).
    pub fn to_zarr_value(&self, dtype: DataType) -> ZarrValue {
        match self {
            FillValue::Value(v) if v.data_type() == dtype => v.clone(),
            _ => default_scalar(dtype),
        }
    }
}

/// Default zero/false/empty scalar for a data type.
pub fn default_scalar(dtype: DataType) -> ZarrValue {
    match dtype {
        DataType::Bool => ZarrValue::Bool(false),
        DataType::Int8 => ZarrValue::Int8(0),
        DataType::Int16 => ZarrValue::Int16(0),
        DataType::Int32 => ZarrValue::Int32(0),
        DataType::Int64 => ZarrValue::Int64(0),
        DataType::UInt8 => ZarrValue::UInt8(0),
        DataType::UInt16 => ZarrValue::UInt16(0),
        DataType::UInt32 => ZarrValue::UInt32(0),
        DataType::UInt64 => ZarrValue::UInt64(0),
        DataType::Float16 => ZarrValue::Float16(f16::ZERO),
        DataType::Float32 => ZarrValue::Float32(0.0),
        DataType::Float64 => ZarrValue::Float64(0.0),
        DataType::Complex64 => ZarrValue::Complex64(Complex::new(0.0f32, 0.0)),
        DataType::Complex128 => ZarrValue::Complex128(Complex::new(0.0f64, 0.0)),
        DataType::String => ZarrValue::String(std::string::String::new()),
        DataType::Bytes => ZarrValue::Bytes(Vec::new()),
    }
}

/// Default fill value for a data type.
pub fn default_fill_value(dtype: DataType) -> FillValue {
    FillValue::Value(default_scalar(dtype))
}

// ---------------------------------------------------------------------------
// ZarrVectorValue  (typed chunk data)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ZarrVectorValue {
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

impl ZarrVectorValue {
    /// Number of elements in the vector.
    pub fn len(&self) -> usize {
        match self {
            ZarrVectorValue::VBool(v) => v.len(),
            ZarrVectorValue::VInt8(v) => v.len(),
            ZarrVectorValue::VInt16(v) => v.len(),
            ZarrVectorValue::VInt32(v) => v.len(),
            ZarrVectorValue::VInt64(v) => v.len(),
            ZarrVectorValue::VUInt8(v) => v.len(),
            ZarrVectorValue::VUInt16(v) => v.len(),
            ZarrVectorValue::VUInt32(v) => v.len(),
            ZarrVectorValue::VUInt64(v) => v.len(),
            ZarrVectorValue::VFloat16(v) => v.len(),
            ZarrVectorValue::VFloat32(v) => v.len(),
            ZarrVectorValue::VFloat64(v) => v.len(),
            ZarrVectorValue::VComplex64(v) => v.len(),
            ZarrVectorValue::VComplex128(v) => v.len(),
            ZarrVectorValue::VString(v) => v.len(),
            ZarrVectorValue::VBytes(v) => v.len(),
            ZarrVectorValue::VWithNulls(_, v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lossily convert the entire vector to `Vec<f64>`.
    pub fn to_f64_vec(&self) -> ZarrResult<Vec<f64>> {
        match self {
            ZarrVectorValue::VBool(v) => Ok(v.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect()),
            ZarrVectorValue::VInt8(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VInt16(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VInt32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VInt64(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VUInt8(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VUInt16(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VUInt32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VUInt64(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VFloat16(v) => Ok(v.iter().map(|x| x.to_f64()).collect()),
            ZarrVectorValue::VFloat32(v) => Ok(v.iter().map(|x| *x as f64).collect()),
            ZarrVectorValue::VFloat64(v) => Ok(v.clone()),
            ZarrVectorValue::VComplex64(v) => Ok(v.iter().map(|c| c.re as f64).collect()),
            ZarrVectorValue::VComplex128(v) => Ok(v.iter().map(|c| c.re).collect()),
            ZarrVectorValue::VString(_) => Err(ZarrError::TypeConversion(
                "Cannot convert String to f64".into(),
            )),
            ZarrVectorValue::VBytes(_) => Err(ZarrError::TypeConversion(
                "Cannot convert Bytes to f64".into(),
            )),
            ZarrVectorValue::VWithNulls(_, v) => Ok(v
                .iter()
                .map(|opt| opt.as_ref().and_then(|zv| zv.to_f64()).unwrap_or(f64::NAN))
                .collect()),
        }
    }

    /// Convert to `Vec<Option<ZarrValue>>`, wrapping each element.
    pub fn to_maybe_values(&self) -> Vec<Option<ZarrValue>> {
        match self {
            ZarrVectorValue::VBool(v) => v.iter().map(|x| Some(ZarrValue::Bool(*x))).collect(),
            ZarrVectorValue::VInt8(v) => v.iter().map(|x| Some(ZarrValue::Int8(*x))).collect(),
            ZarrVectorValue::VInt16(v) => v.iter().map(|x| Some(ZarrValue::Int16(*x))).collect(),
            ZarrVectorValue::VInt32(v) => v.iter().map(|x| Some(ZarrValue::Int32(*x))).collect(),
            ZarrVectorValue::VInt64(v) => v.iter().map(|x| Some(ZarrValue::Int64(*x))).collect(),
            ZarrVectorValue::VUInt8(v) => v.iter().map(|x| Some(ZarrValue::UInt8(*x))).collect(),
            ZarrVectorValue::VUInt16(v) => v.iter().map(|x| Some(ZarrValue::UInt16(*x))).collect(),
            ZarrVectorValue::VUInt32(v) => v.iter().map(|x| Some(ZarrValue::UInt32(*x))).collect(),
            ZarrVectorValue::VUInt64(v) => v.iter().map(|x| Some(ZarrValue::UInt64(*x))).collect(),
            ZarrVectorValue::VFloat16(v) => {
                v.iter().map(|x| Some(ZarrValue::Float16(*x))).collect()
            }
            ZarrVectorValue::VFloat32(v) => {
                v.iter().map(|x| Some(ZarrValue::Float32(*x))).collect()
            }
            ZarrVectorValue::VFloat64(v) => {
                v.iter().map(|x| Some(ZarrValue::Float64(*x))).collect()
            }
            ZarrVectorValue::VComplex64(v) => {
                v.iter().map(|x| Some(ZarrValue::Complex64(*x))).collect()
            }
            ZarrVectorValue::VComplex128(v) => {
                v.iter().map(|x| Some(ZarrValue::Complex128(*x))).collect()
            }
            ZarrVectorValue::VString(v) => v
                .iter()
                .map(|x| Some(ZarrValue::String(x.clone())))
                .collect(),
            ZarrVectorValue::VBytes(v) => v
                .iter()
                .map(|x| Some(ZarrValue::Bytes(x.clone())))
                .collect(),
            ZarrVectorValue::VWithNulls(_, v) => v.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw bytes -> typed vector
// ---------------------------------------------------------------------------

/// Interpret raw bytes as a typed vector according to `endian` and `dtype`.
pub fn bytes_to_zarr_vector(
    endian: Endian,
    dtype: DataType,
    data: &[u8],
) -> ZarrResult<ZarrVectorValue> {
    match dtype {
        DataType::Bool => Ok(ZarrVectorValue::VBool(
            data.iter().map(|b| *b != 0).collect(),
        )),
        DataType::Int8 => Ok(ZarrVectorValue::VInt8(
            data.iter().map(|b| *b as i8).collect(),
        )),
        DataType::UInt8 => Ok(ZarrVectorValue::VUInt8(data.to_vec())),

        DataType::Int16 => read_vec_typed(
            endian,
            data,
            |c| c.read_i16::<LittleEndian>(),
            |c| c.read_i16::<BigEndian>(),
            ZarrVectorValue::VInt16,
        ),
        DataType::Int32 => read_vec_typed(
            endian,
            data,
            |c| c.read_i32::<LittleEndian>(),
            |c| c.read_i32::<BigEndian>(),
            ZarrVectorValue::VInt32,
        ),
        DataType::Int64 => read_vec_typed(
            endian,
            data,
            |c| c.read_i64::<LittleEndian>(),
            |c| c.read_i64::<BigEndian>(),
            ZarrVectorValue::VInt64,
        ),
        DataType::UInt16 => read_vec_typed(
            endian,
            data,
            |c| c.read_u16::<LittleEndian>(),
            |c| c.read_u16::<BigEndian>(),
            ZarrVectorValue::VUInt16,
        ),
        DataType::UInt32 => read_vec_typed(
            endian,
            data,
            |c| c.read_u32::<LittleEndian>(),
            |c| c.read_u32::<BigEndian>(),
            ZarrVectorValue::VUInt32,
        ),
        DataType::UInt64 => read_vec_typed(
            endian,
            data,
            |c| c.read_u64::<LittleEndian>(),
            |c| c.read_u64::<BigEndian>(),
            ZarrVectorValue::VUInt64,
        ),

        DataType::Float16 => {
            let elem_size = 2;
            let count = data.len() / elem_size;
            let mut out = Vec::with_capacity(count);
            let mut cursor = Cursor::new(data);
            for _ in 0..count {
                let bits = match endian {
                    Endian::Little | Endian::NotApplicable => cursor.read_u16::<LittleEndian>(),
                    Endian::Big => cursor.read_u16::<BigEndian>(),
                }
                .map_err(|e| ZarrError::Decode(format!("Failed to read f16: {e}")))?;
                out.push(f16::from_bits(bits));
            }
            Ok(ZarrVectorValue::VFloat16(out))
        }
        DataType::Float32 => read_vec_typed(
            endian,
            data,
            |c| c.read_f32::<LittleEndian>(),
            |c| c.read_f32::<BigEndian>(),
            ZarrVectorValue::VFloat32,
        ),
        DataType::Float64 => read_vec_typed(
            endian,
            data,
            |c| c.read_f64::<LittleEndian>(),
            |c| c.read_f64::<BigEndian>(),
            ZarrVectorValue::VFloat64,
        ),

        DataType::Complex64 => {
            let elem_size = 8;
            let count = data.len() / elem_size;
            let mut out = Vec::with_capacity(count);
            let mut cursor = Cursor::new(data);
            for _ in 0..count {
                let re = match endian {
                    Endian::Little | Endian::NotApplicable => cursor.read_f32::<LittleEndian>(),
                    Endian::Big => cursor.read_f32::<BigEndian>(),
                }
                .map_err(|e| ZarrError::Decode(format!("Failed to read complex64 re: {e}")))?;
                let im = match endian {
                    Endian::Little | Endian::NotApplicable => cursor.read_f32::<LittleEndian>(),
                    Endian::Big => cursor.read_f32::<BigEndian>(),
                }
                .map_err(|e| ZarrError::Decode(format!("Failed to read complex64 im: {e}")))?;
                out.push(Complex::new(re, im));
            }
            Ok(ZarrVectorValue::VComplex64(out))
        }
        DataType::Complex128 => {
            let elem_size = 16;
            let count = data.len() / elem_size;
            let mut out = Vec::with_capacity(count);
            let mut cursor = Cursor::new(data);
            for _ in 0..count {
                let re = match endian {
                    Endian::Little | Endian::NotApplicable => cursor.read_f64::<LittleEndian>(),
                    Endian::Big => cursor.read_f64::<BigEndian>(),
                }
                .map_err(|e| ZarrError::Decode(format!("Failed to read complex128 re: {e}")))?;
                let im = match endian {
                    Endian::Little | Endian::NotApplicable => cursor.read_f64::<LittleEndian>(),
                    Endian::Big => cursor.read_f64::<BigEndian>(),
                }
                .map_err(|e| ZarrError::Decode(format!("Failed to read complex128 im: {e}")))?;
                out.push(Complex::new(re, im));
            }
            Ok(ZarrVectorValue::VComplex128(out))
        }
        DataType::String | DataType::Bytes => Err(ZarrError::Decode(
            "Cannot interpret raw bytes as String/Bytes vector without length info".into(),
        )),
    }
}

/// Helper: read a vector of a fixed-size numeric type.
fn read_vec_typed<T: Clone, F1, F2>(
    endian: Endian,
    data: &[u8],
    read_le: F1,
    read_be: F2,
    wrap: fn(Vec<T>) -> ZarrVectorValue,
) -> ZarrResult<ZarrVectorValue>
where
    F1: Fn(&mut Cursor<&[u8]>) -> std::io::Result<T>,
    F2: Fn(&mut Cursor<&[u8]>) -> std::io::Result<T>,
{
    let elem_size = std::mem::size_of::<T>();
    let count = data.len() / elem_size;
    let mut out = Vec::with_capacity(count);
    let mut cursor = Cursor::new(data);
    for _ in 0..count {
        let val = match endian {
            Endian::Little | Endian::NotApplicable => (read_le)(&mut cursor),
            Endian::Big => (read_be)(&mut cursor),
        }
        .map_err(|e| ZarrError::Decode(format!("Failed to read value: {e}")))?;
        out.push(val);
    }
    Ok(wrap(out))
}

/// Create a filled chunk vector by replicating a scalar value.
pub fn fill_chunk(value: &ZarrValue, chunk_shape: &[usize]) -> ZarrVectorValue {
    let total: usize = chunk_shape.iter().product();
    match value {
        ZarrValue::Bool(b) => ZarrVectorValue::VBool(vec![*b; total]),
        ZarrValue::Int8(v) => ZarrVectorValue::VInt8(vec![*v; total]),
        ZarrValue::Int16(v) => ZarrVectorValue::VInt16(vec![*v; total]),
        ZarrValue::Int32(v) => ZarrVectorValue::VInt32(vec![*v; total]),
        ZarrValue::Int64(v) => ZarrVectorValue::VInt64(vec![*v; total]),
        ZarrValue::UInt8(v) => ZarrVectorValue::VUInt8(vec![*v; total]),
        ZarrValue::UInt16(v) => ZarrVectorValue::VUInt16(vec![*v; total]),
        ZarrValue::UInt32(v) => ZarrVectorValue::VUInt32(vec![*v; total]),
        ZarrValue::UInt64(v) => ZarrVectorValue::VUInt64(vec![*v; total]),
        ZarrValue::Float16(v) => ZarrVectorValue::VFloat16(vec![*v; total]),
        ZarrValue::Float32(v) => ZarrVectorValue::VFloat32(vec![*v; total]),
        ZarrValue::Float64(v) => ZarrVectorValue::VFloat64(vec![*v; total]),
        ZarrValue::Complex64(v) => ZarrVectorValue::VComplex64(vec![*v; total]),
        ZarrValue::Complex128(v) => ZarrVectorValue::VComplex128(vec![*v; total]),
        ZarrValue::String(s) => ZarrVectorValue::VString(vec![s.clone(); total]),
        ZarrValue::Bytes(b) => ZarrVectorValue::VBytes(vec![b.clone(); total]),
        ZarrValue::Null(dt) => ZarrVectorValue::VWithNulls(*dt, vec![None; total]),
    }
}
