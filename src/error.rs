//! Error types for the inference bridge.
use crate::engine::EngineType;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// The serialized engine file could not be read or deserialized.
    #[error("failed to load engine at {path}: {source}")]
    Load { path: String, #[source] source: std::io::Error },

    #[error("backend {engine_type} is not implemented yet")]
    UnsupportedBackend { engine_type: EngineType },

     #[error("failed to load native bridge {path}: {message}")]
    BridgeLoad { path: String, message: String },

    #[error("bridge error: {message}")]
    Engine { message: String },
}

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

impl From<DataError> for InferError {
    fn from(err: DataError) -> Self {
        InferError::Backend {
            message: format!("data error: {}", err),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DataError {
    // The DataStructure cannot be created from the given data, e.g. a tensor shape is invalid.
    #[error("invalid data structure: {datatype} - {message}")]
    InvalidStructure { datatype: String, message: String },
}

pub type InferResult<T> = Result<T, InferError>;
pub type EngineResult<T> = Result<T, EngineError>;
pub type DataResult<T> = Result<T, DataError>;
