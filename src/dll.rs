//! Runtime loading of the C++ bridge DLL (`infertensorrt_cpp.dll`).
//!
//! The DLL is *not* linked at build time: `build.rs` only generates the type
//! declarations in `crate::ffi`, and every exported function is resolved here
//! with `libloading`. The generated `extern "C"` declarations in `crate::ffi`
//! are therefore unused — call the pointers on [`Bridge`] instead.
//!
//! ```no_run
//! use rsinference::dll::Bridge;
//!
//! let bridge = Bridge::load_default_backend().unwrap();
//! unsafe {
//!     let mut error = rsinference::dll::new_error();
//!     let runner = (bridge.runner_create)(
//!         b"demo.engine\0".as_ptr() as *const std::ffi::c_char,
//!         std::ptr::null(),
//!         &mut error,
//!     );
//!     if !runner.is_null() {
//!         (bridge.runner_destroy)(runner);
//!     }
//! }
//! ```
//!
//! Keep the [`Bridge`] alive for the whole program: dropping it unloads the DLL
//! and invalidates every function pointer it handed out.

use libloading::Library;
use std::os::raw::{c_char, c_int, c_uint};
use std::path::{Path, PathBuf};

use crate::error::{EngineError, EngineResult, InferError, InferResult};
use crate::ffi;

// ---------------------------------------------------------------------------
// Helpers on the bindgen-generated C structs (definitions live in crate::ffi,
// generated from cpp/include/public/rsinference.h).
// ---------------------------------------------------------------------------

impl ffi::RSInferLLArray {
    pub unsafe fn as_slice(&self) -> &[i64] {
        if self.data.is_null() || self.length == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data as *const i64, self.length)
        }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &mut [i64] {
        if self.data.is_null() || self.length == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(self.data as *mut i64, self.length)
        }
    }
}

impl ffi::RSInferFPArray {
    pub unsafe fn as_slice(&self) -> &[f32] {
        if self.data.is_null() || self.length == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data as *const f32, self.length)
        }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.data.is_null() || self.length == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(self.data as *mut f32, self.length)
        }
    }
}

impl ffi::RSInferUCharArray {
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.data.is_null() || self.length == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data as *const u8, self.length)
        }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.data.is_null() || self.length == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(self.data as *mut u8, self.length)
        }
    }
}

impl ffi::RSInferIntArray {
    pub unsafe fn as_slice(&self) -> &[i32] {
        if self.data.is_null() || self.length == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(self.data as *const i32, self.length)
        }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &mut [i32] {
        if self.data.is_null() || self.length == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(self.data as *mut i32, self.length)
        }
    }
}

// ---------------------------------------------------------------------------
// Function-pointer types, mirroring cpp/include/public/rsinference.h exactly.
// ---------------------------------------------------------------------------

type FnLlarrayCreate = unsafe extern "C" fn(length: usize) -> *mut ffi::RSInferLLArray;
type FnLlarrayDestroy = unsafe extern "C" fn(array: *mut ffi::RSInferLLArray);

type FnFparrayCreate = unsafe extern "C" fn(length: usize) -> *mut ffi::RSInferFPArray;
type FnFparrayDestroy = unsafe extern "C" fn(array: *mut ffi::RSInferFPArray);

type FnUchararrayCreate = unsafe extern "C" fn(length: usize) -> *mut ffi::RSInferUCharArray;
type FnUchararrayDestroy = unsafe extern "C" fn(array: *mut ffi::RSInferUCharArray);

type FnIntarrayCreate = unsafe extern "C" fn(length: usize) -> *mut ffi::RSInferIntArray;
type FnIntarrayDestroy = unsafe extern "C" fn(array: *mut ffi::RSInferIntArray);

type FnErrorCleanmsg = unsafe extern "C" fn(error: *mut ffi::RSInferError);

type FnTensorCreate = unsafe extern "C" fn(
    name: *const c_char,
    name_length: usize,
    dtype: ffi::RSInferDataType,
    data: *mut ffi::RSInferUCharArray,
    shape: *mut ffi::RSInferLLArray,
) -> *mut ffi::RSInferTensor;
type FnTensorDestroy = unsafe extern "C" fn(tensor: *mut ffi::RSInferTensor);

type FnTensorarrayCreate = unsafe extern "C" fn(length: usize) -> *mut ffi::RSInferTensorArray;
type FnTensorarrayDestroy = unsafe extern "C" fn(array: *mut ffi::RSInferTensorArray);

type FnRunnerCreate = unsafe extern "C" fn(
    model_path: *const c_char,
    config_path: *const c_char,
    error: *mut ffi::RSInferError,
) -> *mut ffi::RSInferenceRunner;
type FnRunnerDestroy = unsafe extern "C" fn(runner: *mut ffi::RSInferenceRunner);
type FnRunnerInfer = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    input_tensors: *mut ffi::RSInferTensorArray,
    error: *mut ffi::RSInferError,
) -> c_int;
type FnRunnerGetResults = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    error: *mut ffi::RSInferError,
) -> *mut ffi::RSInferTensorArray;
type FnRunnerGetShape = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    index: c_uint,
    error: *mut ffi::RSInferError,
) -> *mut ffi::RSInferIntArray;
type FnRunnerGetTensorDataType = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    tensor_name: *const c_char,
    error: *mut ffi::RSInferError,
) -> ffi::RSInferDataType;
type FnRunnerGetTensorDataSize = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    data_type: ffi::RSInferDataType,
    error: *mut ffi::RSInferError,
) -> usize;
type FnRunnerGetTensorName = unsafe extern "C" fn(
    runner: *mut ffi::RSInferenceRunner,
    index: c_uint,
    error: *mut ffi::RSInferError,
) -> *const c_char;

/// Every exported symbol of the bridge, resolved once at load time.
///
/// Calling a field is unsafe (`unsafe { (bridge.runner_infer)(...) }`): the
/// pointers themselves are checked, but the arguments and the C++ side effects
/// are your responsibility.
pub struct Bridge {
    /// Keeps the DLL loaded; must outlive every call made through the fields.
    lib: Library,
    /// Path the DLL was loaded from, for diagnostics.
    pub path: PathBuf,

    pub llarray_create: FnLlarrayCreate,
    pub llarray_destroy: FnLlarrayDestroy,

    pub fparray_create: FnFparrayCreate,
    pub fparray_destroy: FnFparrayDestroy,

    pub uchararray_create: FnUchararrayCreate,
    pub uchararray_destroy: FnUchararrayDestroy,

    pub intarray_create: FnIntarrayCreate,
    pub intarray_destroy: FnIntarrayDestroy,

    pub error_cleanmsg: FnErrorCleanmsg,

    pub tensor_create: FnTensorCreate,
    pub tensor_destroy: FnTensorDestroy,

    pub tensorarray_create: FnTensorarrayCreate,
    pub tensorarray_destroy: FnTensorarrayDestroy,

    pub runner_create: FnRunnerCreate,
    pub runner_destroy: FnRunnerDestroy,
    pub runner_infer: FnRunnerInfer,
    pub runner_infer_async: FnRunnerInfer,
    pub runner_get_results: FnRunnerGetResults,
    pub runner_get_input_shape: FnRunnerGetShape,
    pub runner_get_output_shape: FnRunnerGetShape,
    pub runner_get_tensor_data_type: FnRunnerGetTensorDataType,
    pub runner_get_tensor_data_size: FnRunnerGetTensorDataSize,
    pub runner_get_input_tensor_name: FnRunnerGetTensorName,
    pub runner_get_output_tensor_name: FnRunnerGetTensorName,
}

/// Raw function pointers are not `Debug`, so this reports the loaded DLL only.
impl std::fmt::Debug for Bridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// Name of the CMake target that produces the bridge, matching `build.rs`.
const DLL_NAME: &str = "infertensorrt_cpp";

/// File name the loader looks for on the current platform.
fn library_file_name(backend: &str) -> String {
    format!(
        "{}{}{}",
        std::env::consts::DLL_PREFIX,
        backend,
        std::env::consts::DLL_SUFFIX
    )
}

/// Resolve one exported symbol and copy the function pointer out of the
/// `Symbol`, so the struct never borrows its own `Library` field.
unsafe fn lookup<T: Copy>(lib: &Library, name: &[u8], path: &str) -> EngineResult<T> {
    lib.get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|source| EngineError::BridgeLoad {
            path: path.to_string(),
            message: format!(
                "missing symbol {} ({source})",
                String::from_utf8_lossy(name)
            ),
        })
}

impl Bridge {
    /// Load the bridge DLL from `path` and resolve every exported symbol.
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Self> {
        let path = path.as_ref().to_path_buf();
        let display = path.display().to_string();

        let lib = unsafe { Library::new(&path) }.map_err(|source| EngineError::BridgeLoad {
            path: display.clone(),
            message: source.to_string(),
        })?;

        let display = display.as_str();
        unsafe {
            Ok(Self {
                llarray_create: lookup(&lib, b"rs_infer_llarray_create", display)?,
                llarray_destroy: lookup(&lib, b"rs_infer_llarray_destroy", display)?,
                fparray_create: lookup(&lib, b"rs_infer_fparray_create", display)?,
                fparray_destroy: lookup(&lib, b"rs_infer_fparray_destroy", display)?,
                uchararray_create: lookup(&lib, b"rs_infer_uchararray_create", display)?,
                uchararray_destroy: lookup(&lib, b"rs_infer_uchararray_destroy", display)?,
                intarray_create: lookup(&lib, b"rs_infer_intarray_create", display)?,
                intarray_destroy: lookup(&lib, b"rs_infer_intarray_destroy", display)?,
                error_cleanmsg: lookup(&lib, b"rs_infer_error_cleanmsg", display)?,
                tensor_create: lookup(&lib, b"rs_infer_tensor_create", display)?,
                tensor_destroy: lookup(&lib, b"rs_infer_tensor_destroy", display)?,
                tensorarray_create: lookup(&lib, b"rs_infer_tensorarray_create", display)?,
                tensorarray_destroy: lookup(&lib, b"rs_infer_tensorarray_destroy", display)?,
                runner_create: lookup(&lib, b"rs_inference_runner_create", display)?,
                runner_destroy: lookup(&lib, b"rs_inference_runner_destroy", display)?,
                runner_infer: lookup(&lib, b"rs_inference_runner_infer", display)?,
                runner_infer_async: lookup(&lib, b"rs_inference_runner_infer_async", display)?,
                runner_get_results: lookup(&lib, b"rs_inference_runner_get_results", display)?,
                runner_get_input_shape: lookup(
                    &lib,
                    b"rs_inference_runner_get_input_shape",
                    display,
                )?,
                runner_get_output_shape: lookup(
                    &lib,
                    b"rs_inference_runner_get_output_shape",
                    display,
                )?,
                runner_get_tensor_data_type: lookup(
                    &lib,
                    b"rs_inference_runner_get_tensor_data_type",
                    display,
                )?,
                runner_get_tensor_data_size: lookup(
                    &lib,
                    b"rs_inference_runner_get_tensor_data_size",
                    display,
                )?,
                runner_get_input_tensor_name: lookup(
                    &lib,
                    b"rs_inference_runner_get_input_tensor_name",
                    display,
                )?,
                runner_get_output_tensor_name: lookup(
                    &lib,
                    b"rs_inference_runner_get_output_tensor_name",
                    display,
                )?,
                path,
                lib,
            })
        }
    }

    /// Try the executable's directory, then a bare name resolved through the
    /// platform search order (`PATH`).
    ///
    /// Nothing machine-specific is baked into the binary: a published build
    /// works wherever the bridge happens to be installed.
    pub fn load_default_backend() -> EngineResult<Self> {
        Self::load_backend(DLL_NAME)
    }

    /// Like [`Bridge::load_default_backend`], but for the bridge named `backend`
    /// (e.g. `"inferonnx_cpp"`).
    pub fn load_backend(backend: &str) -> EngineResult<Self> {
        let name = library_file_name(backend);
        let mut candidates: Vec<PathBuf> = Vec::new();

        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join(&name));
            }
        }
        candidates.push(PathBuf::from(&name));

        let mut last = EngineError::BridgeLoad {
            path: name.clone(),
            message: "no candidate path existed".to_string(),
        };
        for candidate in candidates {
            let is_bare_name = candidate == PathBuf::from(&name);
            if !candidate.exists() && !is_bare_name {
                last = EngineError::BridgeLoad {
                    path: candidate.display().to_string(),
                    message: "file not found".to_string(),
                };
                continue;
            }
            match Self::load(&candidate) {
                Ok(bridge) => return Ok(bridge),
                Err(err) => last = err,
            }
        }
        Err(last)
    }

    /// Release the DLL handle (`Library::close`). After this call every
    /// function pointer on the struct is dangling, so only call it when the
    /// bridge is truly done; dropping a `Bridge` unloads the DLL too.
    pub fn unload(self) -> EngineResult<()> {
        let path = self.path.display().to_string();
        self.lib.close().map_err(|source| EngineError::BridgeLoad {
            path,
            message: source.to_string(),
        })?;
        Ok(())
    }
}

/// A zeroed `RSInferError` ready to hand to the C++ side.
///
/// The header's C++ default member initializers (`int code = 0; char* message =
/// NULL;`) only apply when C++ constructs the struct; a Rust-side struct must
/// be initialized explicitly.
pub fn new_error() -> ffi::RSInferError {
    ffi::RSInferError {
        code: 0,
        message: std::ptr::null_mut(),
    }
}

/// Read an `RSInferError` written by the bridge into a Rust error, freeing the
/// message through `rs_infer_error_cleanmsg`.
pub fn take_error(error: &mut ffi::RSInferError, bridge: &Bridge) -> Option<InferError> {
    if error.code == 0 && error.message.is_null() {
        return None;
    }
    let message = if error.message.is_null() {
        format!("native error code {}", error.code)
    } else {
        unsafe { std::ffi::CStr::from_ptr(error.message) }
            .to_string_lossy()
            .into_owned()
    };
    unsafe { (bridge.error_cleanmsg)(error) };
    error.code = 0;
    error.message = std::ptr::null_mut();
    Some(InferError::Backend { message })
}

/// `c_int` result -> `Result<()>`, folding in the error struct when non-zero.
pub fn check(code: c_int, error: &mut ffi::RSInferError, bridge: &Bridge) -> InferResult<()> {
    if code == 0 {
        return Ok(());
    }
    match take_error(error, bridge) {
        Some(err) => Err(err),
        None => Err(InferError::Backend {
            message: format!("native call failed with code {code}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // needs the bridge DLL plus its TensorRT/CUDA dependencies reachable
    fn load_default_resolves_every_symbol() {
        let bridge = Bridge::load_default_backend().expect("bridge DLL should load");
        assert!(bridge.path.exists());
    }

    #[test]
    fn missing_dll_reports_bridge_load() {
        let err = Bridge::load("definitely-not-there.dll").unwrap_err();
        assert!(matches!(err, EngineError::BridgeLoad { .. }));
    }

    #[test]
    fn new_error_is_zeroed() {
        let error = new_error();
        assert_eq!(error.code, 0);
        assert!(error.message.is_null());
    }
}