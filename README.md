# rsinference

`rsinference` is a Rust API over a native inference bridge. The Rust crate
loads the bridge library at runtime, while the bridge connects to the backend
runtime. The repository currently implements the TensorRT bridge.

```text
Rust application → rsinference → infertensorrt shared library → TensorRT → GPU
```

## Project status

| Backend | Bridge | Status |
| --- | --- | --- |
| TensorRT | `infertensorrt` | Implemented: runner creation, tensor metadata, sync/async inference, result retrieval |
| ONNX Runtime | `inferonnx_cpp` | Not implemented |
| OpenVINO | `inferopenvino_cpp` | Not implemented |

`ONNX` models are currently supported through the TensorRT backend's ONNX
parser. The separate ONNX Runtime backend is not available yet.

## Requirements

Building the Rust crate requires:

- Rust and Cargo
- C++ bridge header at `cpp/include/public/rsinference.h` (included here)
- `libclang`/LLVM available to `bindgen`
- A previously built `infertensorrt` shared library for runtime use

Building the bridge requires CMake, a C++23 compiler, Ninja (for the presets),
vcpkg dependencies, CUDA Toolkit, and TensorRT. The CMake project also requires
OpenCV, Eigen3, and GoogleTest. The vendor SDKs and runtime libraries are not
bundled with the Rust crate.

## Build the native bridge

The repository includes Windows and Linux CMake presets in
[`cpp/CMakePresets.json`](cpp/CMakePresets.json). The presets expect CUDA,
TensorRT, and the vcpkg toolchain at the paths configured in that file; adjust
them for your machine as needed. For example, from a Developer PowerShell on
Windows:

```powershell
cmake --preset x64-release -S cpp
cmake --build cpp/build/x64-release
```

The Rust build script searches for the bridge in `cpp/build/x64-release` and
copies it beside the Rust executable. To use a bridge built elsewhere, set
`INFERTENSORRT_BIN_DIR` to the directory containing `infertensorrt.dll` (or the
platform's `.so`/`.dylib`) before building.

## Build and run the Rust CLI

```powershell
cargo build
cargo run
```

The current CLI loads the TensorRT bridge and prints the selected backend. It
does not accept a model path or run inference; use the library API or the
face-detection example to create a runner and perform inference.

## Library API

Load the implemented backend, then create an inference instance for a model:

```rust
use std::path::PathBuf;
use rsinference::{Engine, EngineType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::load(EngineType::TensorRT)?;
    let model = PathBuf::from("models/version-RFB-640.onnx");
    let runner = engine.create_infer_instance(&model, None)?;

    println!("backend: {}", runner.engine_type());
    println!("input: {} {:?}", runner.input_name(0)?, runner.input_shape(0)?);
    println!("output: {} {:?}", runner.output_name(0)?, runner.output_shape(0)?);
    Ok(())
}
```

`create_infer_instance` accepts an optional backend JSON configuration string.
`InferInstance` also provides `tensor_dtype`, synchronous `infer`,
`infer_async`, and `get_results`. Inputs and outputs use the public
`rsinference::types::Tensor` and `Dtype` types. For a complete working example
that preprocesses images and consumes inference results, see
[`examples/facedetection`](examples/facedetection/README.md).

Tensor shapes use signed `i64` dimensions so dynamic or unresolved dimensions
reported by a backend (such as `-1`) are preserved.

## Native library lookup

The bridge is loaded at runtime. `Bridge::load_default_backend()` checks next
to the current executable, then asks the platform loader to resolve the bare
library name (typically through `PATH`). You can also load a library by path
with `Bridge::load(path)` or by name with `Bridge::load_backend(name)`.

TensorRT's own DLLs/shared libraries must also be discoverable by the process.
Set `PATH` (Windows) or the platform's library search path before starting the
application. On Windows, TensorRT and CUDA runtime DLLs usually need to be on
`PATH` or beside the executable; placing them only beside the bridge DLL is
not sufficient. Match the bridge, application, vendor runtime, and system
architecture.

## Face detection example

[`examples/facedetection`](examples/facedetection/README.md) is a webcam demo
using OpenCV and the included UltraFace RFB 640 ONNX model. It demonstrates
preprocessing, TensorRT inference, result handling, and drawing detections.
See [`models/README.md`](models/README.md) for model details and the example
README for build and runtime setup.

## Errors

The public error types are `EngineError` (bridge loading and runner creation),
`InferError` (inference and tensor operations), and `DataError` (malformed
native tensor data). They are non-exhaustive; downstream matches should include
a catch-all arm. Requests for ONNX Runtime or OpenVINO currently return
`UnsupportedBackend`.

## License

MIT
