//! Engine handling shared by all backends.

use std::cell::OnceCell;
use std::ffi::{CStr, CString, c_int};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::dll::take_error;
use crate::error::{DataResult, EngineError, EngineResult, InferError, InferResult};
use crate::ffi::{RSInferError, RSInferTensor, RSInferTensorArray, RSInferenceRunner};
use crate::Bridge;
use crate::types::*;
use std::os::raw::c_char;

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
    ///
    /// The OpenVINO bridge is only built when CMake finds OpenVINO, so on a
    /// machine without it [`Engine::load`] still reports
    /// [`EngineError::BridgeLoad`] for that backend.
    pub const fn is_implemented(self) -> bool {
        matches!(self, EngineType::TensorRT | EngineType::OpenVINO)
    }

    /// CMake target name of the C++ bridge that implements this backend.
    ///
    /// [`Bridge::load_backend`] turns this into `infertensorrt.dll`,
    /// `libinfertensorrt.so`, `libinfertensorrt.dylib`, ... for the
    /// platform the crate was built for.
    const fn bridge_name(self) -> &'static str {
        match self {
            EngineType::TensorRT => "infertensorrt",
            EngineType::OpenVINO => "inferopenvino",
            EngineType::ONNX => "inferonnx",
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

#[derive(Debug)]
pub struct InferInstance<'a> {
    runner: *mut RSInferenceRunner,
    engine: &'a Engine,
    model_path: PathBuf,
    /// A built runner's inputs and outputs never change, so each list is
    /// fetched from the bridge once, on first use.
    inputs: OnceCell<Vec<TensorInfo>>,
    outputs: OnceCell<Vec<TensorInfo>>,
}

// Identity is the runner pointer only. Deriving would require Bridge: PartialEq/Eq.
impl PartialEq for InferInstance<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.runner == other.runner
    }
}
impl Eq for InferInstance<'_> {}

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
    /// Name, shape and data type of every model input, in model order.
    /// Dynamic dimensions are `-1`; an empty shape means even the rank is unknown.
    pub fn inputs_info(&self) -> InferResult<Vec<TensorInfo>> {
        self.cached_inputs().map(<[TensorInfo]>::to_vec)
    }

    /// Name, shape and data type of every model output, in model order.
    /// Dynamic dimensions are `-1`; an empty shape means even the rank is unknown.
    pub fn outputs_info(&self) -> InferResult<Vec<TensorInfo>> {
        self.cached_outputs().map(<[TensorInfo]>::to_vec)
    }

    fn cached_inputs(&self) -> InferResult<&[TensorInfo]> {
        self.cached(&self.inputs, self.engine.bridge.runner_get_input_tensors_info, "runner_get_input_tensors_info")
    }

    fn cached_outputs(&self) -> InferResult<&[TensorInfo]> {
        self.cached(&self.outputs, self.engine.bridge.runner_get_output_tensors_info, "runner_get_output_tensors_info")
    }

    /// Returns `cell`'s list, fetching it through `fetch` the first time.
    /// A failed fetch is not cached, so a later call retries.
    fn cached<'s>(
        &self,
        cell: &'s OnceCell<Vec<TensorInfo>>,
        fetch: crate::dll::FnRunnerGetTensorsInfo,
        call: &str,
    ) -> InferResult<&'s [TensorInfo]> {
        if let Some(infos) = cell.get() {
            return Ok(infos);
        }
        let mut err = RSInferError::default();
        let ptr = unsafe { fetch(self.runner, &mut err) };
        let infos = self
            .take_tensor_array(ptr, &mut err, call)?
            .into_iter()
            .map(TensorInfo::from)
            .collect();
        Ok(cell.get_or_init(|| infos))
    }

    /// The runner's effective configuration as a JSON string. It uses the same
    /// keys as the `json_config` passed to [`Engine::create_infer_instance`], so
    /// it can be passed back to create an equivalent instance.
    pub fn json_config(&self) -> InferResult<String> {
        let mut err = RSInferError::default();
        let ptr = unsafe { (self.engine.bridge.runner_get_json_config)(self.runner, &mut err) };
        if ptr.is_null() {
            return Err(take_error(&mut err, &self.engine.bridge).unwrap_or(InferError::Backend {
                message: "runner_get_json_config returned null".to_string(),
            }));
        }
        let config = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        // The bridge allocated this; free it with the bridge's C runtime.
        unsafe { (self.engine.bridge.string_free)(ptr) };
        if let Some(error) = take_error(&mut err, &self.engine.bridge) {
            return Err(error);
        }
        Ok(config)
    }

    /// Info of the input at `order`.
    pub fn input_info(&self, order: u32) -> InferResult<TensorInfo> {
        nth_info(self.cached_inputs()?, order, "input").cloned()
    }

    /// Info of the output at `order`.
    pub fn output_info(&self, order: u32) -> InferResult<TensorInfo> {
        nth_info(self.cached_outputs()?, order, "output").cloned()
    }

    /// Name of the input tensor, as expected by the engine.
    pub fn input_name(&self, order: u32) -> InferResult<String> {
        Ok(self.input_info(order)?.name)
    }

    /// Name of the output tensor, as expected by the engine.
    pub fn output_name(&self, order: u32) -> InferResult<String> {
        Ok(self.output_info(order)?.name)
    }

    /// Data type of the input or output tensor called `name`.
    pub fn tensor_dtype(&self, name: &str) -> InferResult<Dtype> {
        self.cached_inputs()?
            .iter()
            .chain(self.cached_outputs()?)
            .find(|info| info.name == name)
            .map(|info| info.dtype)
            .ok_or_else(|| InferError::Backend { message: format!("no tensor named {name:?}") })
    }

    /// Converts a bridge-allocated tensor array to Rust and frees it. Handles the
    /// null-on-error and error-alongside-result cases of every array-returning call.
    fn take_tensor_array(
        &self,
        ptr: *mut RSInferTensorArray,
        err: &mut RSInferError,
        call: &str,
    ) -> InferResult<Vec<Tensor>> {
        let bridge = &self.engine.bridge;
        if ptr.is_null() {
            return Err(take_error(err, bridge).unwrap_or(InferError::Backend {
                message: format!("{call} returned null"),
            }));
        }
        if let Some(error) = take_error(err, bridge) {
            unsafe { (bridge.tensorarray_destroy)(ptr) };
            return Err(error);
        }

        let tensors = unsafe { tensors_from_array(&*ptr, call) };

        // Release the array even if a tensor fails conversion.
        unsafe { (bridge.tensorarray_destroy)(ptr) };
        tensors.map_err(InferError::from)
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
        let tensors = unsafe { (*rs_infer_tensor_arr_ptr).as_mut_slice() };

        for (raw_t, inp) in tensors.iter_mut().zip(inputs.iter()) {
            let rs_tensor_ptr = match unsafe { self.rs_tensor_from_tensor(inp) } {
                Ok(ptr) => ptr,
                Err(err) => {
                    unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
                    return Err(err);
                }
            };
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
        let tensors = unsafe { (*rs_infer_tensor_arr_ptr).as_mut_slice() };

        for (raw_t, inp) in tensors.iter_mut().zip(inputs.iter()) {
            let rs_tensor_ptr = match unsafe { self.rs_tensor_from_tensor(inp) } {
                Ok(ptr) => ptr,
                Err(err) => {
                    unsafe { (self.engine.bridge.tensorarray_destroy)(rs_infer_tensor_arr_ptr) };
                    return Err(err);
                }
            };
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
        let ptr = unsafe { (self.engine.bridge.runner_get_results)(self.runner, &mut err) };
        self.take_tensor_array(ptr, &mut err, "runner_get_results")
    }

    /// Input shape of the engine, in row-major order. Dynamic dimensions are `-1`.
    pub fn input_shape(&self, order: u32) -> InferResult<Shape> {
        Ok(self.input_info(order)?.shape)
    }

    /// Output shape of the engine, in row-major order. Dynamic dimensions are `-1`.
    pub fn output_shape(&self, order: u32) -> InferResult<Shape> {
        Ok(self.output_info(order)?.shape)
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
        let ll_slice = unsafe { (*rs_ll_arr_ptr).as_mut_slice() };
        for (item, dim) in ll_slice.iter_mut().zip(tensor.shape.iter()) {
            *item = *dim;
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
        let data_slice = unsafe { (*data_ptr).as_mut_slice() };
        for (item, dat) in data_slice.iter_mut().zip(tensor.data.iter()) {
            *item = dat.clone();
        }
        // Create tensor array
        let rs_tensor_ptr = unsafe { (self.engine.bridge.tensor_create)(name_cstr, name_len, dtype_c as _, data_ptr, rs_ll_arr_ptr) };
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

/// Copies every tensor out of a bridge-owned array. A zero-length array may have
/// a NULL tensor list; a non-zero length with a NULL list is malformed.
unsafe fn tensors_from_array(array: &RSInferTensorArray, call: &str) -> DataResult<Vec<Tensor>> {
    if array.length > 0 && array.tensors.is_null() {
        return Err(crate::error::DataError::InvalidStructure {
            datatype: "tensor array".to_string(),
            message: format!("{call} returned {} tensors but a null tensor list", array.length),
        });
    }
    array
        .as_slice()
        .iter()
        .map(|t| Tensor::from_rs_infer_tensor_ptr(t))
        .collect()
}

fn nth_info<'i>(infos: &'i [TensorInfo], order: u32, kind: &str) -> InferResult<&'i TensorInfo> {
    let count = infos.len();
    infos.get(order as usize).ok_or_else(|| InferError::Backend {
        message: format!("{kind} index {order} is out of range; the model has {count} {kind}(s)"),
    })
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

    pub fn create_infer_instance(&self, model_path: &PathBuf, json_config: Option<&str>) -> EngineResult<InferInstance<'_>> {

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

        let model_c = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            EngineError::Load {
                path: model_path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "path contains an interior NUL",
                ),
            }
        })?;

        let config_c = json_config
            .map(CString::new)
            .transpose()
            .map_err(|_| EngineError::Engine {
                message: "json config contains an interior NUL".to_string(),
            })?;

        let mut error = RSInferError::default();
        let runner = unsafe {
            (self.bridge.runner_create)(
                model_c.as_ptr(),
                config_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
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
            inputs: OnceCell::new(),
            outputs: OnceCell::new(),
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

    /// Declares a test that needs real hardware. It is ignored (and listed as such) unless
    /// the `hardware-tests` feature is set: `cargo test --features hardware-tests`.
    macro_rules! hardware_test {
        ($(#[$meta:meta])* fn $name:ident() $body:block) => {
            #[test]
            #[cfg_attr(
                not(feature = "hardware-tests"),
                ignore = "needs real hardware; run with --features hardware-tests"
            )]
            $(#[$meta])*
            fn $name() $body
        };
    }

    hardware_test! {
        fn load_missing_model_reports_error() {
            let Some(engine) = tensorrt_engine() else { return };
            let err = engine.create_infer_instance(&PathBuf::from("nonexistent.engine"), None).unwrap_err();
            assert!(matches!(err, EngineError::Load { .. }));
        }
    }

    #[test]
    fn tensor_array_with_null_list_is_rejected() {
        let malformed = RSInferTensorArray { tensors: ptr::null_mut(), length: 2 };
        let err = unsafe { tensors_from_array(&malformed, "runner_get_results") }.unwrap_err();
        assert!(err.to_string().contains("2 tensors but a null tensor list"), "{err}");
    }

    #[test]
    fn empty_tensor_array_may_have_null_list() {
        let empty = RSInferTensorArray { tensors: ptr::null_mut(), length: 0 };
        assert!(unsafe { tensors_from_array(&empty, "runner_get_results") }.unwrap().is_empty());
    }

    #[test]
    fn unimplemented_backend_is_reported() {
        let err = Engine::load(EngineType::ONNX).unwrap_err();
        assert!(matches!(err, EngineError::UnsupportedBackend { .. }));
    }

    // Needs an NVIDIA GPU to build the TensorRT engine.
    hardware_test! {
        fn load_existing_model_succeeds() {
            let engine = Engine::load(EngineType::TensorRT).unwrap();
            let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/version-RFB-640.onnx");
            let instance = engine.create_infer_instance(&model_path, None).unwrap();
            assert_eq!(instance.model_path(), &model_path);
        }
    }

    // OpenVINO runs the sample model on the CPU, so these need no GPU. See
    // models/README.md for the expected tensor names and shapes.

    const OPENVINO_CPU: Option<&str> = Some(r#"{"device": "CPU"}"#);

    /// Loads a backend's bridge. Outside CI, a missing runtime skips the test instead of
    /// failing it; in CI (where `CI` is set) it is a hard failure.
    fn load_or_skip(backend: EngineType, hint: &str) -> Option<Engine> {
        match Engine::load(backend) {
            Ok(engine) => Some(engine),
            Err(EngineError::BridgeLoad { message, .. }) if std::env::var_os("CI").is_none() => {
                eprintln!("skipping: {backend:?} bridge did not load ({message}); {hint}");
                None
            }
            Err(err) => panic!("{backend:?} bridge did not load: {err:?}"),
        }
    }

    fn tensorrt_engine() -> Option<Engine> {
        load_or_skip(
            EngineType::TensorRT,
            "put TensorRT's bin and the CUDA toolkit's bin on PATH (LD_LIBRARY_PATH on Linux)",
        )
    }

    fn openvino_engine() -> Option<Engine> {
        load_or_skip(
            EngineType::OpenVINO,
            r"put OpenVINO's runtime\bin\intel64\Release and runtime\3rdparty\tbb\bin on PATH",
        )
    }

    fn sample_model() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/version-RFB-640.onnx")
    }

    /// A zero-filled `1 x 3 x 480 x 640` float32 input for the sample model.
    fn sample_input() -> Tensor {
        Tensor {
            name: "input".to_string(),
            shape: vec![1, 3, 480, 640],
            dtype: Dtype::F32,
            data: vec![0u8; 3 * 480 * 640 * 4],
        }
    }

    fn as_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// Checks the outputs of one inference of the sample model.
    fn assert_sample_outputs(outputs: &[Tensor]) {
        assert_eq!(outputs.len(), 2);
        let scores = outputs.iter().find(|t| t.name == "scores").expect("scores output");
        let boxes = outputs.iter().find(|t| t.name == "boxes").expect("boxes output");

        assert_eq!(scores.shape, vec![1, 17640, 2]);
        assert_eq!(boxes.shape, vec![1, 17640, 4]);
        assert_eq!(scores.dtype, Dtype::F32);
        assert_eq!(boxes.dtype, Dtype::F32);
        assert_eq!(scores.data.len(), 17640 * 2 * 4);
        assert_eq!(boxes.data.len(), 17640 * 4 * 4);

        // The graph ends in a softmax over [background, face].
        for pair in as_f32(&scores.data).chunks_exact(2) {
            assert!((pair[0] + pair[1] - 1.0).abs() < 1e-3, "scores {pair:?} do not sum to 1");
        }
    }

    hardware_test! {
        fn openvino_load_missing_model_reports_error() {
            let Some(engine) = openvino_engine() else { return };
            assert_eq!(engine.engine_type(), EngineType::OpenVINO);
            let err = engine.create_infer_instance(&PathBuf::from("nonexistent.onnx"), OPENVINO_CPU).unwrap_err();
            assert!(matches!(err, EngineError::Load { .. }));
        }
    }

    hardware_test! {
        fn openvino_rejects_invalid_config() {
            let Some(engine) = openvino_engine() else { return };
            let err = engine.create_infer_instance(&sample_model(), Some(r#"{"device": 5}"#)).unwrap_err();
            match err {
                EngineError::Engine { message } => assert!(message.contains("device must be a string"), "{message}"),
                other => panic!("expected EngineError::Engine, got {other:?}"),
            }

            let err = engine.create_infer_instance(&sample_model(), Some("not json")).unwrap_err();
            assert!(matches!(err, EngineError::Engine { .. }));
        }
    }

    hardware_test! {
        fn openvino_reports_model_io() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            assert_eq!(instance.engine_type(), EngineType::OpenVINO);

            assert_eq!(instance.input_name(0).unwrap(), "input");
            assert_eq!(instance.input_shape(0).unwrap(), vec![1, 3, 480, 640]);
            assert_eq!(instance.tensor_dtype("input").unwrap(), Dtype::F32);

            let mut outputs = [
                (instance.output_name(0).unwrap(), instance.output_shape(0).unwrap()),
                (instance.output_name(1).unwrap(), instance.output_shape(1).unwrap()),
            ];
            outputs.sort();
            assert_eq!(outputs, [
                ("boxes".to_string(), vec![1, 17640, 4]),
                ("scores".to_string(), vec![1, 17640, 2]),
            ]);
            assert_eq!(instance.tensor_dtype("scores").unwrap(), Dtype::F32);

            assert!(instance.input_name(1).is_err());
            assert!(instance.output_shape(2).is_err());
        }
    }

    /// Checks `inputs_info`/`outputs_info` against the sample model.
    fn assert_sample_info(instance: &InferInstance<'_>) {
        assert_eq!(instance.inputs_info().unwrap(), vec![TensorInfo {
            name: "input".to_string(),
            shape: vec![1, 3, 480, 640],
            dtype: Dtype::F32,
        }]);
        let mut outputs = instance.outputs_info().unwrap();
        outputs.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(outputs, vec![
            TensorInfo { name: "boxes".to_string(), shape: vec![1, 17640, 4], dtype: Dtype::F32 },
            TensorInfo { name: "scores".to_string(), shape: vec![1, 17640, 2], dtype: Dtype::F32 },
        ]);
        assert!(instance.tensor_dtype("nope").is_err());
    }

    hardware_test! {
        fn openvino_reports_tensors_info() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            assert_sample_info(&instance);
        }
    }

    hardware_test! {
        fn openvino_json_config_round_trips() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            let config = instance.json_config().unwrap();
            // The bridge emits compact JSON (nlohmann `dump()` without indent).
            assert!(config.contains(r#""device":"CPU""#), "{config}");
            // The reported config is accepted back by create.
            let again = engine.create_infer_instance(&sample_model(), Some(&config)).unwrap();
            assert_eq!(again.json_config().unwrap(), config);
        }
    }

    hardware_test! {
        fn tensorrt_reports_tensors_info() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), None).unwrap();
            assert_sample_info(&instance);
        }
    }

    hardware_test! {
        fn tensorrt_input_tensor_names_are_used_and_checked() {
            let Some(engine) = tensorrt_engine() else { return };
            let config = r#"{"input_tensor_names": ["input"], "output_tensor_names": ["scores", "boxes"]}"#;
            let instance = engine.create_infer_instance(&sample_model(), Some(config)).unwrap();
            assert_sample_info(&instance);
            let names: Vec<_> = instance.outputs_info().unwrap().into_iter().map(|i| i.name).collect();
            assert_eq!(names, ["scores", "boxes"], "outputs follow output_tensor_names order");

            // A name the engine does not have is rejected when the runner is created.
            let err = engine.create_infer_instance(&sample_model(), Some(r#"{"input_tensor_names": ["imput"]}"#));
            assert!(matches!(err, Err(EngineError::Engine { .. })), "{err:?}");
        }
    }

    hardware_test! {
        fn tensorrt_profile_gets_its_own_engine_cache() {
            let Some(engine) = tensorrt_engine() else { return };
            // The sample input is static, so the only valid range is its own shape.
            let config = r#"{"profile": {"input": {"min": [1,3,480,640], "opt": [1,3,480,640], "max": [1,3,480,640]}}}"#;
            let instance = engine.create_infer_instance(&sample_model(), Some(config)).unwrap();
            instance.infer(&[sample_input()]).unwrap();
            assert_sample_outputs(&instance.get_results().unwrap());

            let models = sample_model().parent().unwrap().to_path_buf();
            let cached: Vec<_> = std::fs::read_dir(&models).unwrap()
                .filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter(|n| n.starts_with("version-RFB-640.dynamic-") && n.ends_with(".engine"))
                .collect();
            assert!(!cached.is_empty(), "no profile-keyed engine cache in {}", models.display());
        }
    }

    hardware_test! {
        fn tensorrt_json_config_round_trips() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), Some(r#"{"n_contexts": 2}"#)).unwrap();
            let config = instance.json_config().unwrap();
            // The bridge emits compact JSON (nlohmann `dump()` without indent).
            assert!(config.contains(r#""n_contexts":2"#), "{config}");
            let again = engine.create_infer_instance(&sample_model(), Some(&config)).unwrap();
            assert_eq!(again.json_config().unwrap(), config);
        }
    }

    // These run real inference, which hangs on runners without usable hardware/drivers.
    hardware_test! {
        fn openvino_infer_returns_outputs() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            instance.infer(&[sample_input()]).unwrap();
            assert_sample_outputs(&instance.get_results().unwrap());
        }
    }

    hardware_test! {
        fn openvino_infer_async_returns_outputs() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            instance.infer_async(&[sample_input()]).unwrap();
            instance.infer_async(&[sample_input()]).unwrap();
            assert_sample_outputs(&instance.get_results().unwrap());
            assert_sample_outputs(&instance.get_results().unwrap());
        }
    }

    hardware_test! {
        fn openvino_get_results_without_infer_fails() {
            let Some(engine) = openvino_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), OPENVINO_CPU).unwrap();
            assert!(instance.get_results().is_err());
        }
    }

    // TensorRT builds (or loads the cached) engine for the sample model, which needs an
    // NVIDIA GPU and driver. The first run builds `models/version-RFB-640.engine`.
    hardware_test! {
        fn tensorrt_reports_model_io() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), None).unwrap();
            assert_eq!(instance.engine_type(), EngineType::TensorRT);

            assert_eq!(instance.input_name(0).unwrap(), "input");
            assert_eq!(instance.input_shape(0).unwrap(), vec![1, 3, 480, 640]);
            assert_eq!(instance.tensor_dtype("input").unwrap(), Dtype::F32);

            let mut outputs = [
                (instance.output_name(0).unwrap(), instance.output_shape(0).unwrap()),
                (instance.output_name(1).unwrap(), instance.output_shape(1).unwrap()),
            ];
            outputs.sort();
            assert_eq!(outputs, [
                ("boxes".to_string(), vec![1, 17640, 4]),
                ("scores".to_string(), vec![1, 17640, 2]),
            ]);
            assert_eq!(instance.tensor_dtype("scores").unwrap(), Dtype::F32);

            assert!(instance.input_name(1).is_err());
            assert!(instance.output_shape(2).is_err());
        }
    }

    hardware_test! {
        fn tensorrt_infer_returns_outputs() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), None).unwrap();
            instance.infer(&[sample_input()]).unwrap();
            assert_sample_outputs(&instance.get_results().unwrap());
        }
    }

    hardware_test! {
        fn tensorrt_infer_async_returns_outputs() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), None).unwrap();
            instance.infer_async(&[sample_input()]).unwrap();
            instance.infer_async(&[sample_input()]).unwrap();
            assert_sample_outputs(&instance.get_results().unwrap());
            assert_sample_outputs(&instance.get_results().unwrap());
        }
    }

    hardware_test! {
        fn tensorrt_get_results_without_infer_fails() {
            let Some(engine) = tensorrt_engine() else { return };
            let instance = engine.create_infer_instance(&sample_model(), None).unwrap();
            assert!(instance.get_results().is_err());
        }
    }
}
