use crate::ffi::{RSInferIntArray, RSInferLLArray, RSInferTensor};
use crate::error::{DataResult};

/// Tensor dimensions. Negative values represent dynamic or unresolved dimensions.
pub type Shape = Vec<i64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Default, Ord, Hash)]
pub enum Dtype {
    F32 = 0,
    I32 = 1,
    I64 = 2,
    U8 = 3,
    Bool = 4,
    F16 = 5,
    F8 = 6,
    E8M0 = 7,
    BF16 = 8,
    I4 = 9,
    F4 = 10,
    #[default]
    UNKNOWN = 11,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Tensor {
    pub name: String,
    pub shape: Shape,
    pub dtype: Dtype,
    pub data: Vec<u8>, // Placeholder for actual tensor data
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
            _ => Dtype::UNKNOWN,
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I32 => 4,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
            Dtype::Bool => 1,
            Dtype::F16 => 2,
            Dtype::F8 => 1,
            Dtype::E8M0 => 1,
            Dtype::BF16 => 2,
            Dtype::I4 => 1, // Assuming packed into a byte
            Dtype::F4 => 1, // Assuming packed into a byte
            Dtype::UNKNOWN => 0,
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
            tensor.dtype = Dtype::from_c_int(rs_tensor_ref.dtype);

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
