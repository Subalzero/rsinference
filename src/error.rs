//! Error types for the TensorRT infer bridge.

use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// The serialized engine file could not be read or deserialized.
    #[error("failed to load engine at {path}: {source}")]
    EngineLoad {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// TensorRT returned a failure while building or running the engine.
    #[error("tensorrt error: {message}")]
    TensorRt { message: String },

    /// The input tensor shape does not match the engine's expected shape.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// An operation requires the `ffi` feature to be enabled.
    #[error("native TensorRT bridge not compiled (enable the `ffi` feature)")]
    FfiUnavailable,

    /// The C++ bridge DLL could not be loaded, or one of its symbols is missing.
    #[error("failed to load native bridge {path}: {message}")]
    BridgeLoad { path: String, message: String },
}

pub type InferResult<T> = Result<T, InferError>;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    I32,
    U8,
}
