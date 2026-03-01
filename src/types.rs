use half::f16;
use num_complex::Complex;
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
