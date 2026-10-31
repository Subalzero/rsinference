//! Engine handling shared by all backends.

use std::path::{Path, PathBuf};

use crate::error::{InferError, InferResult};
use crate::Bridge;

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

/// Directory holding the per-backend bridge libraries.
const DEFAULT_BACKEND_PATH: &str = "backends";

/// Candidate bridge directories, in priority order:
///
/// 1. `RSINFER_BACKEND_PATH` at runtime,
/// 2. `RSINFER_BACKEND_PATH` baked in at build time,
/// 3. `<executable dir>/backends` (the reliable location for installed apps),
/// 4. `backends` relative to the current working directory (dev convenience).
fn backend_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Some(dir) = std::env::var_os("RSINFER_BACKEND_PATH") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(dir) = option_env!("RSINFER_BACKEND_PATH") {
        let dir = PathBuf::from(dir);
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let dir = parent.join(DEFAULT_BACKEND_PATH);
            if !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    let cwd = PathBuf::from(DEFAULT_BACKEND_PATH);
    if !dirs.contains(&cwd) {
        dirs.push(cwd);
    }

    dirs
}

/// Load the bridge for `engine_type`, trying every candidate directory.
fn load_bridge(engine_type: EngineType) -> InferResult<Bridge> {
    let mut last = None;
    for dir in backend_dirs() {
        match Bridge::load_backend(&dir, engine_type.bridge_name()) {
            Ok(bridge) => return Ok(bridge),
            Err(err) => last = Some(err),
        }
    }
    Err(last.unwrap_or_else(|| InferError::BridgeLoad {
        path: engine_type.bridge_name().to_string(),
        message: "no backend directory candidates".to_string(),
    }))
}

/// A loaded engine and its native backend bridge.
#[derive(Debug)]
pub struct Engine {
    path: PathBuf,
    engine_type: EngineType,
    /// Must outlive every call made through it; declared last so it drops after
    /// the fields above.
    bridge: Bridge,
}

impl Engine {
    /// Load a serialized engine from disk and open its backend bridge.
    pub fn load(path: impl Into<PathBuf>, engine_type: EngineType) -> InferResult<Self> {
        let path = path.into();

        // Report unimplemented backends as such, before blaming the filesystem.
        if !engine_type.is_implemented() {
            return Err(InferError::UnsupportedBackend { engine_type });
        }

        let metadata = std::fs::metadata(&path).map_err(|source| InferError::EngineLoad {
            path: path.display().to_string(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(InferError::EngineLoad {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "path is not a regular file",
                ),
            });
        }

        let bridge = load_bridge(engine_type)?;

        Ok(Self {
            path,
            engine_type,
            bridge,
        })
    }

    /// Path of the model file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Backend that runs the model.
    pub fn engine_type(&self) -> EngineType {
        self.engine_type
    }

    /// The backend bridge loaded for this engine.
    pub fn bridge(&self) -> &Bridge {
        &self.bridge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_engine_reports_error() {
        let err = Engine::load("definitely-not-an.engine", EngineType::TensorRT).unwrap_err();
        assert!(matches!(err, InferError::EngineLoad { .. }));
    }

    #[test]
    fn unimplemented_backend_is_reported() {
        let err = Engine::load("model.engine", EngineType::ONNX).unwrap_err();
        assert!(matches!(err, InferError::UnsupportedBackend { .. }));
    }
}
