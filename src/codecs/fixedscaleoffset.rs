use crate::error::{ZarrError, ZarrResult};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedScaleOffsetCodec {
    pub scale: f64,
    pub offset: f64,
    pub dtype: String,
    pub astype: String,
}

impl FixedScaleOffsetCodec {
    pub fn decode(&self, data: &[u8]) -> ZarrResult<Vec<u8>> {
        match (self.astype.as_str(), self.dtype.as_str()) {
            ("int16", "float32") => self.decode_int_to_float::<i16>(data, 2),
            ("int32", "float32") => self.decode_int_to_float::<i32>(data, 4),
            ("uint16", "float32") => self.decode_uint_to_float::<u16>(data, 2),
            ("uint32", "float32") => self.decode_uint_to_float::<u32>(data, 4),
            (a, d) => Err(ZarrError::Decode(format!(
                "Unsupported FixedScaleOffset conversion: {a} -> {d}"
            ))),
        }
    }

    pub fn encode(&self, _data: &[u8]) -> ZarrResult<Vec<u8>> {
        Err(ZarrError::Encode(
            "FixedScaleOffsetCodec encoding not implemented".into(),
        ))
    }

    fn decode_int_to_float<T>(&self, data: &[u8], elem_bytes: usize) -> ZarrResult<Vec<u8>>
    where
        T: ReadableInt,
    {
        let count = data.len() / elem_bytes;
        let mut cursor = Cursor::new(data);
        let mut out = Vec::with_capacity(count * 4);
        let mut writer = std::io::Cursor::new(&mut out);
        for _ in 0..count {
            let ival = T::read_le(&mut cursor)
                .map_err(|e| ZarrError::Decode(format!("FixedScaleOffset read: {e}")))?;
            let fval = ival.to_f64() * self.scale + self.offset;
            writer
                .write_f32::<LittleEndian>(fval as f32)
                .map_err(|e| ZarrError::Decode(format!("FixedScaleOffset write: {e}")))?;
        }
        Ok(out)
    }

    fn decode_uint_to_float<T>(&self, data: &[u8], elem_bytes: usize) -> ZarrResult<Vec<u8>>
    where
        T: ReadableUInt,
    {
        let count = data.len() / elem_bytes;
        let mut cursor = Cursor::new(data);
        let mut out = Vec::with_capacity(count * 4);
        let mut writer = std::io::Cursor::new(&mut out);
        for _ in 0..count {
            let uval = T::read_le(&mut cursor)
                .map_err(|e| ZarrError::Decode(format!("FixedScaleOffset read: {e}")))?;
            let fval = uval.to_f64() * self.scale + self.offset;
            writer
                .write_f32::<LittleEndian>(fval as f32)
                .map_err(|e| ZarrError::Decode(format!("FixedScaleOffset write: {e}")))?;
        }
        Ok(out)
    }
}

trait ReadableInt {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self>
    where
        Self: Sized;
    fn to_f64(self) -> f64;
}

impl ReadableInt for i16 {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        cursor.read_i16::<LittleEndian>()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ReadableInt for i32 {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        cursor.read_i32::<LittleEndian>()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

trait ReadableUInt {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self>
    where
        Self: Sized;
    fn to_f64(self) -> f64;
}

impl ReadableUInt for u16 {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        cursor.read_u16::<LittleEndian>()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ReadableUInt for u32 {
    fn read_le(cursor: &mut Cursor<&[u8]>) -> std::io::Result<Self> {
        cursor.read_u32::<LittleEndian>()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
