//! Error types for the inference bridge.

use std::fmt;

use crate::engine::EngineType;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferError {
    /// The serialized engine file could not be read or deserialized.
    #[error("failed to load engine at {path}: {source}")]
    EngineLoad {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The backend reported a failure while building or running the engine.
    #[error("backend error: {message}")]
    Backend { message: String },

    /// The input tensor shape does not match the engine's expected shape.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// The C++ bridge library could not be loaded, or one of its symbols is
    /// missing.
    #[error("failed to load native bridge {path}: {message}")]
    BridgeLoad { path: String, message: String },

    /// The requested backend has no native bridge in this build.
    #[error("backend {engine_type} is not implemented yet")]
    UnsupportedBackend { engine_type: EngineType },
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
