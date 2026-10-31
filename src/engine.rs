//! TensorRT engine handling.

use crate::error::{Dtype, InferError, InferResult, Shape};

/// A loaded (or to-be-loaded) TensorRT engine.
#[derive(Debug)]
pub struct Engine {
    /// Path to the serialized `.engine` file on disk.
    pub path: std::path::PathBuf,
    /// Input tensor the engine expects.
    pub input: Shape,
    /// Output tensor the engine produces.
    pub output: Shape,
}

impl Engine {
    /// Load a serialized engine from disk.
    pub fn load(path: impl Into<std::path::PathBuf>) -> InferResult<Self> {
        let path = path.into();
        let _bytes = std::fs::read(&path).map_err(|source| InferError::EngineLoad {
            path: path.display().to_string(),
            source,
        })?;

        // TODO: with the `ffi` feature, drive the C++ runner instead of reading
        // bytes here: Bridge::load_default(), then
        // (bridge.runner_create)(model_ptr, config_ptr, &mut error) from
        // `crate::dll`.
        let input = Shape {
            dims: Vec::new(),
            dtype: Dtype::F32,
        };
        let output = Shape {
            dims: Vec::new(),
            dtype: Dtype::F32,
        };

        Ok(Self {
            path,
            input,
            output,
        })
    }

    /// Run inference on `input`. Length must match the engine's input shape.
    pub fn infer(&self, input: &[f32]) -> InferResult<Vec<f32>> {
        let actual = input.len();
        let expected = self.input.dims.iter().product::<usize>();
        if expected > 0 && actual != expected {
            return Err(InferError::ShapeMismatch {
                expected: expected.to_string(),
                actual: actual.to_string(),
            });
        }

        if !cfg!(feature = "ffi") {
            return Err(InferError::FfiUnavailable);
        }

        // TODO: call the runtime-resolved pointers in `crate::dll::Bridge`:
        // runner_infer / runner_get_results on the FFI surface declared in
        // cpp/include/public/rsinference.h.
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_engine_reports_error() {
        let err = Engine::load("definitely-not-an.engine").unwrap_err();
        assert!(matches!(err, InferError::EngineLoad { .. }));
    }

    #[cfg(not(feature = "ffi"))]
    #[test]
    fn infer_requires_ffi_feature() {
        let engine = Engine {
            path: std::path::PathBuf::from("demo.engine"),
            input: Shape {
                dims: vec![2],
                dtype: Dtype::F32,
            },
            output: Shape {
                dims: vec![2],
                dtype: Dtype::F32,
            },
        };
        assert!(matches!(
            engine.infer(&[1.0, 2.0]),
            Err(InferError::FfiUnavailable)
        ));
    }
}
