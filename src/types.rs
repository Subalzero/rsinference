use std::fmt;

use crate::ffi::{RSInferIntArray, RSInferLLArray, RSInferTensor};
use crate::error::{DataResult};

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        write!(f, "{}", dims)
    }
}

/// Tensor shape: dimensions in row-major order, plus element kind.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape {
    pub dims: Vec<u64>,
}

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

impl Shape {
    /// Changes the c object to Rust object Shape
    /// Make sure to check if the input is not null
    pub unsafe fn from_rs_infer_int_array_ptr(rs_array: &*mut RSInferIntArray) -> DataResult<Self> {
        let rs_int_arr_ref = unsafe { &**rs_array };
        let dims = unsafe { std::slice::from_raw_parts(rs_int_arr_ref.data, rs_int_arr_ref.length as usize) }
            .iter()
            .map(|&d| d as u64)
            .collect::<Vec<_>>();
        Ok(Shape {
            dims,
        })
    }

    /// Changes the c object to Rust object Shape
    /// Make sure to check if the input is not null
    pub unsafe fn from_rs_infer_ll_array_ptr(rs_array: &*mut RSInferLLArray) -> DataResult<Self> {
        let rs_int_arr_ref = unsafe { &**rs_array };
        let dims = unsafe { std::slice::from_raw_parts(rs_int_arr_ref.data, rs_int_arr_ref.length as usize) }
            .iter()
            .map(|&d| d as u64)
            .collect::<Vec<_>>();
        Ok(Shape {
            dims,
        })
    }
}

impl Tensor {
    /// Changes the c object to Rust object Tensor
    /// Make sure to check if the input is not null
    pub unsafe fn from_rs_infer_tensor_ptr(rs_tensor: &*mut RSInferTensor) -> DataResult<Self> {
        let mut tensor = Tensor::default();

        unsafe {
            let rs_tensor_ref = &**rs_tensor;
            tensor.name = std::ffi::CStr::from_ptr(rs_tensor_ref.name)
                .to_string_lossy()
                .into_owned();
            tensor.shape = Shape::from_rs_infer_ll_array_ptr(&rs_tensor_ref.shape)?;
            tensor.dtype = Dtype::from_c_int(rs_tensor_ref.dtype);
        }
        Ok(tensor)
    }
}