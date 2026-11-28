use crate::ffi::{RSInferIntArray, RSInferLLArray, RSInferTensor};
use crate::error::{DataResult};

/// Tensor dimensions. Negative values represent dynamic or unresolved dimensions.
pub type Shape = Vec<i64>;

/// Tensor element type. Discriminants match the C `RSInferDataType` values.
///
/// New types may be added in minor releases, so match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Default, Ord, Hash)]
#[non_exhaustive]
pub enum Dtype {
    F32 = 0,
    I32 = 1,
    I64 = 2,
    U8 = 3,
    Bool = 4,
    F16 = 5,
    /// 8-bit float, E4M3 layout.
    F8 = 6,
    E8M0 = 7,
    BF16 = 8,
    /// Packed 4-bit integer: two elements per byte.
    I4 = 9,
    /// Packed 4-bit float: two elements per byte.
    F4 = 10,
    I8 = 11,
    /// 8-bit float, E5M2 layout.
    F8E5M2 = 12,
    #[default]
    UNKNOWN = 13,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Tensor {
    pub name: String,
    pub shape: Shape,
    pub dtype: Dtype,
    pub data: Vec<u8>, // Placeholder for actual tensor data
}

/// Description of a model input or output, as reported by the backend.
///
/// Dynamic dimensions are `-1`, and an empty shape means even the rank is
/// unknown. Send fully concrete shapes when you infer.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Shape,
    pub dtype: Dtype,
}

impl TensorInfo {
    /// Whether the shape is not fully known: some dimension is dynamic
    /// (negative), or the shape is empty.
    ///
    /// An empty shape counts as dynamic because the bridges use it for a tensor
    /// whose rank is unknown. A genuine rank-0 scalar also has an empty shape,
    /// so it is reported as dynamic too: the C ABI cannot tell the two apart.
    pub fn is_dynamic(&self) -> bool {
        self.shape.is_empty() || self.shape.iter().any(|&dim| dim < 0)
    }

    /// Whether the number of dimensions is unknown (reported as an empty shape).
    /// Also true for a rank-0 scalar, which the C ABI cannot distinguish.
    pub fn is_rank_unknown(&self) -> bool {
        self.shape.is_empty()
    }
}

impl From<Tensor> for TensorInfo {
    fn from(tensor: Tensor) -> Self {
        TensorInfo { name: tensor.name, shape: tensor.shape, dtype: tensor.dtype }
    }
}

impl Dtype {
    pub fn from_c_int(value: i32) -> Self {
        match value {
            0 => Dtype::F32,
            1 => Dtype::I32,
            2 => Dtype::I64,
            3 => Dtype::U8,
            4 => Dtype::Bool,
            5 => Dtype::F16,
            6 => Dtype::F8,
            7 => Dtype::E8M0,
            8 => Dtype::BF16,
            9 => Dtype::I4,
            10 => Dtype::F4,
            11 => Dtype::I8,
            12 => Dtype::F8E5M2,
            _ => Dtype::UNKNOWN,
        }
    }

    /// Bytes per element. Packed 4-bit types (`I4`, `F4`) report 1, their
    /// one-byte storage unit; use [`Dtype::byte_count`] to size a buffer.
    pub fn byte_size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I32 => 4,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
            Dtype::I8 => 1,
            Dtype::Bool => 1,
            Dtype::F16 => 2,
            Dtype::F8 => 1,
            Dtype::F8E5M2 => 1,
            Dtype::E8M0 => 1,
            Dtype::BF16 => 2,
            Dtype::I4 => 1,
            Dtype::F4 => 1,
            Dtype::UNKNOWN => 0,
        }
    }

    /// Bytes needed to hold `element_count` elements, accounting for 4-bit
    /// packing. Returns 0 for `UNKNOWN`.
    pub fn byte_count(&self, element_count: usize) -> usize {
        match self {
            Dtype::I4 | Dtype::F4 => element_count.div_ceil(2),
            _ => element_count * self.byte_size(),
        }
    }
}

impl Tensor {
    /// Changes the c object to Rust object Tensor
    /// Make sure to check if the input is not null
    pub unsafe fn from_rs_infer_tensor_ptr(rs_tensor: &*mut RSInferTensor) -> DataResult<Self> {
        let mut tensor = Tensor::default();

        if rs_tensor.is_null() {
            return Err(crate::error::DataError::InvalidStructure {
                datatype: "tensor".to_string(),
                message: "received a null tensor pointer".to_string(),
            });
        }

        unsafe {
            let rs_tensor_ref = &**rs_tensor;
            if rs_tensor_ref.name.is_null() {
                return Err(crate::error::DataError::InvalidStructure {
                    datatype: "tensor".to_string(),
                    message: "tensor name is null".to_string(),
                });
            }
            tensor.name = std::ffi::CStr::from_ptr(rs_tensor_ref.name)
                .to_string_lossy()
                .into_owned();
            tensor.shape = from_rs_infer_ll_array_ptr(&rs_tensor_ref.shape)?;
            tensor.dtype = Dtype::from_c_int(rs_tensor_ref.dtype as i32);

            if !rs_tensor_ref.data.is_null() {
                let data = &*rs_tensor_ref.data;
                if !data.data.is_null() {
                    tensor.data =
                        std::slice::from_raw_parts(data.data, data.length).to_vec();
                }
            }
        }
        Ok(tensor)
    }
}

pub unsafe fn from_rs_infer_int_array_ptr(rs_array: &*mut RSInferIntArray) -> DataResult<Shape> {
        if rs_array.is_null() {
            return Err(crate::error::DataError::InvalidStructure {
                datatype: "shape".to_string(),
                message: "received a null int array pointer".to_string(),
            });
        }
        let rs_int_arr_ref = unsafe { &**rs_array };
        if rs_int_arr_ref.data.is_null() {
            return Ok(Vec::new());
        }
        let dims = unsafe { std::slice::from_raw_parts(rs_int_arr_ref.data, rs_int_arr_ref.length as usize) }
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        Ok(dims)
    }

    /// Changes the c object to Rust object Shape
    /// Make sure to check if the input is not null
    pub unsafe fn from_rs_infer_ll_array_ptr(rs_array: &*mut RSInferLLArray) -> DataResult<Shape> {
        if rs_array.is_null() {
            return Err(crate::error::DataError::InvalidStructure {
                datatype: "shape".to_string(),
                message: "received a null shape pointer".to_string(),
            });
        }
        let rs_int_arr_ref = unsafe { &**rs_array };
        if rs_int_arr_ref.data.is_null() {
            return Ok(Vec::new());
        }
        let dims = unsafe { std::slice::from_raw_parts(rs_int_arr_ref.data, rs_int_arr_ref.length as usize) }
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        Ok(dims)
    }

#[cfg(test)]
mod tests {
    use super::Dtype;
    use crate::ffi;

    /// Every Dtype paired with its C constant. Keep in sync with rsinference.h.
    const ALL: &[(Dtype, ffi::RSInferDataType)] = &[
        (Dtype::F32, ffi::RSInferDataType_RSInferDataType_Float32),
        (Dtype::I32, ffi::RSInferDataType_RSInferDataType_Int32),
        (Dtype::I64, ffi::RSInferDataType_RSInferDataType_Int64),
        (Dtype::U8, ffi::RSInferDataType_RSInferDataType_UInt8),
        (Dtype::Bool, ffi::RSInferDataType_RSInferDataType_Bool),
        (Dtype::F16, ffi::RSInferDataType_RSInferDataType_Half),
        (Dtype::F8, ffi::RSInferDataType_RSInferDataType_FP8),
        (Dtype::E8M0, ffi::RSInferDataType_RSInferDataType_E8M0),
        (Dtype::BF16, ffi::RSInferDataType_RSInferDataType_BF16),
        (Dtype::I4, ffi::RSInferDataType_RSInferDataType_Int4),
        (Dtype::F4, ffi::RSInferDataType_RSInferDataType_FP4),
        (Dtype::I8, ffi::RSInferDataType_RSInferDataType_Int8),
        (Dtype::F8E5M2, ffi::RSInferDataType_RSInferDataType_FP8E5M2),
        (Dtype::UNKNOWN, ffi::RSInferDataType_RSInferDataType_Unknown),
    ];

    #[test]
    fn discriminants_match_c_header() {
        for &(dtype, c_value) in ALL {
            assert_eq!(dtype as i32, c_value as i32, "{dtype:?}");
        }
    }

    #[test]
    fn from_c_int_round_trips() {
        for &(dtype, c_value) in ALL {
            assert_eq!(Dtype::from_c_int(c_value as i32), dtype);
        }
        assert_eq!(Dtype::from_c_int(-1), Dtype::UNKNOWN);
        assert_eq!(Dtype::from_c_int(1000), Dtype::UNKNOWN);
    }

    #[test]
    fn byte_sizes() {
        assert_eq!(Dtype::F32.byte_size(), 4);
        assert_eq!(Dtype::I64.byte_size(), 8);
        assert_eq!(Dtype::F16.byte_size(), 2);
        assert_eq!(Dtype::I8.byte_size(), 1);
        assert_eq!(Dtype::F8E5M2.byte_size(), 1);
        assert_eq!(Dtype::I4.byte_size(), 1);
        assert_eq!(Dtype::UNKNOWN.byte_size(), 0);
    }

    #[test]
    fn unknown_rank_and_negative_dims_are_dynamic() {
        use super::TensorInfo;
        let info = |shape: Vec<i64>| TensorInfo { name: "t".to_string(), shape, dtype: Dtype::F32 };
        assert!(!info(vec![1, 3, 480, 640]).is_dynamic());
        assert!(info(vec![-1, 3, 480, 640]).is_dynamic());
        assert!(info(vec![]).is_dynamic());
        assert!(info(vec![]).is_rank_unknown());
        assert!(!info(vec![-1]).is_rank_unknown());
    }

    #[test]
    fn byte_count_packs_4bit_types() {
        assert_eq!(Dtype::F32.byte_count(3), 12);
        assert_eq!(Dtype::I4.byte_count(4), 2);
        assert_eq!(Dtype::I4.byte_count(5), 3);
        assert_eq!(Dtype::F4.byte_count(0), 0);
        assert_eq!(Dtype::UNKNOWN.byte_count(10), 0);
    }
}
