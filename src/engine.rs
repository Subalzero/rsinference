//! Engine handling shared by all backends.

use std::ffi::{CStr, c_int};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::dll::take_error;
use crate::error::{EngineError, EngineResult, InferError, InferResult};
use crate::ffi::{RSInferError, RSInferTensor, RSInferenceRunner};
use crate::Bridge;
use crate::types::*;
use std::os::raw::{c_char, c_uint, c_void};

unsafe extern "C" {
    unsafe fn free(ptr: *mut c_void);
}

/// Inference backend an [`Engine`] talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EngineType {
    TensorRT,
    OpenVINO,
    ONNX,
}

impl EngineType {
    /// Whether the C++ bridge for this backend exists in this repository.
    ///
    /// Backends without a bridge are rejected with
    /// [`InferError::UnsupportedBackend`] instead of a misleading "file not
    /// found" when the loader looks for a library that was never built.
    pub const fn is_implemented(self) -> bool {
        matches!(self, EngineType::TensorRT)
    }

    /// CMake target name of the C++ bridge that implements this backend.
    ///
    /// [`Bridge::load_backend`] turns this into `infertensorrt_cpp.dll`,
    /// `libinfertensorrt_cpp.so`, `libinfertensorrt_cpp.dylib`, ... for the
    /// platform the crate was built for.
    const fn bridge_name(self) -> &'static str {
        match self {
            EngineType::TensorRT => "infertensorrt_cpp",
            EngineType::OpenVINO => "inferopenvino_cpp",
            EngineType::ONNX => "inferonnx_cpp",
        }
    }
}

impl std::fmt::Display for EngineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EngineType::TensorRT => "TensorRT",
            EngineType::OpenVINO => "OpenVINO",
            EngineType::ONNX => "ONNX Runtime",
        })
    }
}

/// A loaded engine and its native backend bridge.
#[derive(Debug)]
pub struct Engine {
    engine_type: EngineType,
    /// Must outlive every call made through it, including `runner`.
    bridge: Bridge,
}

unsafe impl Sync for Engine {}

#[derive(Debug)]
pub struct InferInstance<'a> {
    runner: *mut RSInferenceRunner,
    engine: &'a Engine,
    model_path: PathBuf,
}

// Identity is the runner pointer only. Deriving would require Bridge: PartialEq/Eq.
impl PartialEq for InferInstance<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.runner == other.runner
    }
}
impl Eq for InferInstance<'_> {}

// SAFETY: every `&self` method only calls bridge functions that are
// thread-safe on the C++ side, and the Rust side holds no shared mutable state.
unsafe impl Send for InferInstance<'_> {}
unsafe impl Sync for InferInstance<'_> {}

impl Hash for InferInstance<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.runner as usize).hash(state);
    }
}

impl Drop for InferInstance<'_> {
    fn drop(&mut self) {
        if !self.runner.is_null() {
            // `drop` runs before the fields are dropped, so `bridge` is still
            // loaded and `runner_destroy` is still callable.
            unsafe { (self.engine.bridge.runner_destroy)(self.runner) };
            self.runner = ptr::null_mut();
        }
    }
}

impl InferInstance<'_> {
    /// Name of the input tensor, as expected by the engine.
    pub fn input_name(&self, order: u32) -> Result<String, InferError> {
        let mut err: RSInferError = RSInferError::default();
        let name_ptr = unsafe { (self.engine.bridge.runner_get_input_tensor_name)(self.runner, order as c_uint, &mut err) };

        if name_ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_input_name returned null".to_string(),
            }));
        }

        if err.code != 0 || !err.message.is_null() {
            let error = take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_input_name returned an error".to_string(),
            });
            return Err(error);
        }

        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();

        // The C++ side allocated this with strdup; free it.
        unsafe { free(name_ptr as *mut c_void) };

        Ok(name)
    }

    /// Name of the output tensor, as expected by the engine.
    pub fn output_name(&self, order: u32) -> Result<String, InferError> {
        let mut err: RSInferError = RSInferError::default();
        let name_ptr = unsafe { (self.engine.bridge.runner_get_output_tensor_name)(self.runner, order as c_uint, &mut err) };

        if name_ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_output_name returned null".to_string(),
            }));
        }

        if err.code != 0 || !err.message.is_null() {
            let error = take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_output_name returned an error".to_string(),
            });
            return Err(error);
        }

        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();

        // The C++ side allocated this with strdup; free it.
        unsafe { free(name_ptr as *mut c_void) };

        Ok(name)
    }

    /// Data type of a tensor, as expected by the engine specified by name.
    pub fn tensor_dtype(&self, name: &str) -> Result<Dtype, InferError> {
        let mut err: RSInferError = RSInferError::default();
        let dtype = unsafe { (self.engine.bridge.runner_get_tensor_data_type)(self.runner, name.as_ptr() as *const i8, &mut err) };

        if err.code != 0 || !err.message.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_input_tensor_dtype returned an error".to_string(),
            }));
        }

        Ok(Dtype::from_c_int(dtype))
    }

    /// Infer function
    pub fn infer(&self, inputs: &[Tensor]) -> InferResult<()> {
        // Turn tensors into RSInferTensorArray
        let rs_infer_tensor_arr_ptr = unsafe { (self.engine.bridge.tensorarray_create)(inputs.len()) };
        if rs_infer_tensor_arr_ptr.is_null() {
            return Err(
                InferError::Backend { message: "tensorarray_create failed.".to_string() }
            )
        }
        let tensors = unsafe { std::slice::from_raw_parts_mut((*rs_infer_tensor_arr_ptr).tensors, (*rs_infer_tensor_arr_ptr).length) };

        for (raw_t, inp) in tensors.iter_mut().zip(inputs.iter()) {
            let rs_tensor_ptr = unsafe { self.rs_tensor_from_tensor(inp)? };
            *raw_t = rs_tensor_ptr;
        }

        let mut err = RSInferError::default();
        let ret = unsafe { (self.engine.bridge.runner_infer)(self.runner, rs_infer_tensor_arr_ptr, &mut err) };
        if ret != 0 {
            // The C++ side copied everything, so free the inputs whatever rc was.
            unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_infer returned an error".to_string(),
            }));
        }
        if err.code != 0 || !err.message.is_null() {
            // The C++ side copied everything, so free the inputs whatever rc was.
            unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_infer returned an error".to_string(),
            }));
        }

        // The C++ side copied everything, so free the inputs whatever rc was.
        unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };

        Ok(())
    }

    /// Infer function. Runs asynchronously.
    pub fn infer_async(&self, inputs: &[Tensor]) -> InferResult<()> {
        // Turn tensors into RSInferTensorArray
        let rs_infer_tensor_arr_ptr = unsafe { (self.engine.bridge.tensorarray_create)(inputs.len()) };
        if rs_infer_tensor_arr_ptr.is_null() {
            return Err(
                InferError::Backend { message: "tensorarray_create failed.".to_string() }
            )
        }
        let tensors = unsafe { std::slice::from_raw_parts_mut((*rs_infer_tensor_arr_ptr).tensors, (*rs_infer_tensor_arr_ptr).length) };

        for (raw_t, inp) in tensors.iter_mut().zip(inputs.iter()) {
            let rs_tensor_ptr = unsafe { self.rs_tensor_from_tensor(inp)? };
            *raw_t = rs_tensor_ptr;
        }

        let mut err = RSInferError::default();
        let ret = unsafe { (self.engine.bridge.runner_infer_async)(self.runner, rs_infer_tensor_arr_ptr, &mut err) };
        if ret != 0 {
            // The C++ side copied everything, so free the inputs whatever rc was.
            unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_infer returned an error".to_string(),
            }));
        }
        if err.code != 0 || !err.message.is_null() {
            // The C++ side copied everything, so free the inputs whatever rc was.
            unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_infer returned an error".to_string(),
            }));
        }

        // The C++ side copied everything, so free the inputs whatever rc was.
        unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };

        Ok(())
    }

    pub fn get_results(&self) -> InferResult<Vec<Tensor>> {
        let mut err = RSInferError::default();
        let rs_infer_tensor_arr_ptr = unsafe { (self.engine.bridge.runner_get_results)(self.runner, &mut err) };
        if rs_infer_tensor_arr_ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_results returned null".to_string(),
            }));
        }
        if err.code != 0 || !err.message.is_null() {
            // The C++ side copied everything, so free the inputs whatever rc was.
            unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_results returned an error".to_string(),
            }));
        }

        let tensors = unsafe { std::slice::from_raw_parts((*rs_infer_tensor_arr_ptr).tensors, (*rs_infer_tensor_arr_ptr).length) }
            .iter()
            .map(|&t| unsafe { Tensor::from_rs_infer_tensor_ptr(&t) })
            .collect::<Result<Vec<_>, _>>()?;

        // The C++ side copied everything, so free the inputs whatever rc was.
        unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };

        Ok(tensors)
    }

    /// Input shape of the engine, in row-major order.
    pub fn input_shape(&self, order: u32) -> Result<Shape, InferError> {
        let mut err: RSInferError = RSInferError::default();
        let shape_ptr = unsafe { (self.engine.bridge.runner_get_input_shape)(self.runner, order as c_uint, &mut err) };

        if shape_ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_input_shape returned null".to_string(),
            }));
        }

        if err.code != 0 || !err.message.is_null() {
            let error = take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_input_shape returned an error".to_string(),
            });
            return Err(error);
        }

        let shape_result = unsafe { from_rs_infer_int_array_ptr(&shape_ptr) };

        // The C++ side allocated this with rs_infer_intarray_create; free it
        // whether or not the conversion succeeded.
        unsafe { (self.engine.bridge.intarray_destroy)(shape_ptr) };

        let shape: Shape = shape_result.map_err(|e| InferError::ShapeMismatch {
            expected: "valid shape".to_string(),
            actual: format!("invalid shape from bridge: {}", e),
        })?;

        Ok(shape)
    }

    /// Output shape of the engine, in row-major order.
    pub fn output_shape(&self, order: u32) -> Result<Shape, InferError> {
        let mut err: RSInferError = RSInferError::default();
        let shape_ptr = unsafe { (self.engine.bridge.runner_get_output_shape)(self.runner, order as c_uint, &mut err) };

        if shape_ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_output_shape returned null".to_string(),
            }));
        }

        if err.code != 0 || !err.message.is_null() {
            let error = take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_output_shape returned an error".to_string(),
            });
            return Err(error);
        }

        let shape_result = unsafe { from_rs_infer_int_array_ptr(&shape_ptr) };

        // The C++ side allocated this with rs_infer_intarray_create; free it
        // whether or not the conversion succeeded.
        unsafe { (self.engine.bridge.intarray_destroy)(shape_ptr) };

        let shape: Shape = shape_result.map_err(|e| InferError::ShapeMismatch {
            expected: "valid shape".to_string(),
            actual: format!("invalid shape from bridge: {}", e),
        })?;

        Ok(shape)
    }

    /// Creates an RSInferTensor from rust's implementation of Tensor
    /// You must manually delete this object by calling the bridge's tensorarray_destroy
    unsafe fn rs_tensor_from_tensor(&self, tensor: &Tensor) -> InferResult<*mut RSInferTensor> {
        // Convert name to c_str
        let name_cstr = tensor.name.as_ptr() as *const c_char;
        let name_len = tensor.name.len();
        // Convert dtype to c equivalent
        let dtype_c = tensor.dtype as c_int;
        // Convert shape to c equivalent
        let shape_len = tensor.shape.len();
        let rs_ll_arr_ptr = unsafe { (self.engine.bridge.llarray_create)(shape_len) };
        if rs_ll_arr_ptr.is_null() {
            return Err(InferError::Backend { message: "llarray_create returned null".to_string() })
        }

        // populate the llarray
        let ll_slice = unsafe { std::slice::from_raw_parts_mut((*rs_ll_arr_ptr).data, shape_len) };
        for (item, dim) in ll_slice.iter_mut().zip(tensor.shape.iter()) {
            *item = dim.clone() as i64;
        }

        // Convert data to c_data
        let data_len = tensor.data.len();
        let data_ptr = unsafe { (self.engine.bridge.uchararray_create)(data_len) };
        if data_ptr.is_null() {
            unsafe {
                (self.engine.bridge.llarray_destroy)(rs_ll_arr_ptr);
            }
            return Err(InferError::Backend { message: "uchararray_create returned null".to_string() })
        }
        // Populate data
        let data_slice = unsafe { std::slice::from_raw_parts_mut((*data_ptr).data, data_len) };
        for (item, dat) in data_slice.iter_mut().zip(tensor.data.iter()) {
            *item = dat.clone();
        }
        // Create tensor array
        let rs_tensor_ptr = unsafe { (self.engine.bridge.tensor_create)(name_cstr, name_len, dtype_c, data_ptr, rs_ll_arr_ptr) };
        if rs_tensor_ptr.is_null() {
            unsafe {
                (self.engine.bridge.llarray_destroy)(rs_ll_arr_ptr);
                (self.engine.bridge.uchararray_destroy)(data_ptr);
            }
            return Err(InferError::Backend { message: "tensor_create returned null".to_string() })
        }

        Ok(rs_tensor_ptr)
    }

    /// Path of the model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Backend that runs the model.
    pub fn engine_type(&self) -> EngineType {
        self.engine.engine_type
    }

}

impl Engine {
    /// Load a serialized engine from disk and open its backend bridge.
    pub fn load(engine_type: EngineType) -> EngineResult<Self> {
        // Report unimplemented backends as such, before blaming the filesystem.
        if !engine_type.is_implemented() {
            return Err(EngineError::UnsupportedBackend { engine_type });
        }

        let bridge = Bridge::load_backend(engine_type.bridge_name())?;

        Ok(Self {
            engine_type,
            bridge,
        })
    }

    pub fn create_infer_instance(&self, model_path: &PathBuf, config_path: Option<&PathBuf>) -> EngineResult<InferInstance<'_>> {

        let metadata = std::fs::metadata(&model_path).map_err(|source| EngineError::Load {
            path: model_path.display().to_string(),
            source,
        })?;

        if !metadata.is_file() {
            return Err(EngineError::Load {
                path: model_path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "path is not a regular file",
                ),
            });
        }

        let mut error = RSInferError::default();
        let runner = unsafe {
            (self.bridge.runner_create)(
                model_path.to_str().unwrap().as_ptr() as *const i8,
                config_path.map_or(ptr::null(), |p| p.to_str().unwrap().as_ptr() as *const i8),
                &mut error,
            )
        };

        if runner.is_null() {
            let err = take_error(&mut error, &self.bridge).unwrap_or(InferError::Backend {
                message: "runner_create returned a runner and an error".to_string(),
            });

            return Err(
                EngineError::Engine { message: format!("failed to create runner for model {}: {}", model_path.display(), err) }
            )
        } 

        if error.code != 0 || !error.message.is_null() {
            let err = take_error(&mut error, &self.bridge).unwrap_or(InferError::Backend {
                message: "runner_create returned a runner and an error".to_string(),
            });
            unsafe { (self.bridge.runner_destroy)(runner) };
            return Err(EngineError::Engine { message: format!("failed to create runner for model {}: {}", model_path.display(), err) });
        }


        Ok(InferInstance {
            runner: runner,
            engine: self,
            model_path: model_path.clone(),
        })
    }

    /// Backend that runs the model.
    pub fn engine_type(&self) -> EngineType {
        self.engine_type
    }

}

impl Drop for Engine {
    fn drop(&mut self) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_model_reports_error() {
        let engine = Engine::load(EngineType::TensorRT).unwrap();
        let err = engine.create_infer_instance(&PathBuf::from("nonexistent.engine"), None).unwrap_err();
        assert!(matches!(err, EngineError::Load { .. }));
    }

    #[test]
    fn unimplemented_backend_is_reported() {
        let err = Engine::load(EngineType::ONNX).unwrap_err();
        assert!(matches!(err, EngineError::UnsupportedBackend { .. }));
    }

    #[test]
    fn load_existing_model_succeeds() {
        let engine = Engine::load(EngineType::TensorRT).unwrap();
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/version-RFB-640.onnx");
        let instance = engine.create_infer_instance(&model_path, None).unwrap();
        assert_eq!(instance.model_path(), &model_path);
    }
}
