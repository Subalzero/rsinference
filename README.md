# rsinference

`rsinference` is a Rust API over a native inference bridge. The Rust crate
loads the bridge library at runtime, while the bridge connects to the backend
runtime. The repository implements the TensorRT and OpenVINO bridges.

```text
Rust application → rsinference → infertensorrt shared library  → TensorRT → GPU
                              └→ inferopenvino shared library → OpenVINO → CPU/GPU
```

## Project status

| Backend | Bridge | Status |
| --- | --- | --- |
| TensorRT | `infertensorrt` | Implemented: runner creation, tensor metadata, sync/async inference, result retrieval |
| ONNX Runtime | `inferonnx` | Not implemented |
| OpenVINO | `inferopenvino` | Implemented; built only when CMake finds OpenVINO |

`ONNX` models are currently supported through the TensorRT backend's ONNX
parser. The separate ONNX Runtime backend is not available yet.

## Requirements

Building the Rust crate requires:

- Rust and Cargo
- C++ bridge header at `cpp/include/public/rsinference.h` (included here)
- `libclang`/LLVM available to `bindgen`
- A previously built `infertensorrt` (and optionally `inferopenvino`) shared
  library for runtime use, or a prebuilt one from a [release](#releases)

Building the bridge requires CMake, a C++23 compiler, Ninja (for the presets),
vcpkg dependencies, CUDA Toolkit, and TensorRT. OpenVINO is optional: the
`inferopenvino` bridge is built only when CMake finds it (pass `OPENVINO_DIR`).
The CMake project also requires OpenCV, Eigen3, and GoogleTest. The vendor SDKs
and runtime libraries are not bundled with the Rust crate.

### Installing the native SDKs

`scripts/install-deps.sh` (Ubuntu 24.04) and `scripts/install-deps.ps1`
(Windows) download and install the SDKs the CI uses. Pass component names to
install a subset (`cuda`, `tensorrt`, `openvino`, `toolchain`; default: all):

```bash
bash scripts/install-deps.sh cuda tensorrt openvino
```

```powershell
./scripts/install-deps.ps1 cuda tensorrt openvino
```

They place TensorRT and OpenVINO under `DEPS_DIR` (default `$HOME/deps` on
Linux, `C:\deps` on Windows) and print `CUDA_TOOLKIT_ROOT_DIR`, `TENSORRT_DIR`,
`OPENVINO_DIR`, and `LIBCLANG_PATH` for the build. Versions can be overridden
through environment variables; run either script with `-h`/`Get-Help` for the
list. The Windows CUDA installer needs admin rights, and MSVC (Visual Studio
Build Tools) is not installed by the script.

## Build C++ and Rust (Windows)

The C++ bridge is built with CMake; Cargo builds the Rust crate and generates
its FFI bindings. `cargo build` by itself does not build the bridge. The
PowerShell helper `scripts/build.ps1` runs both steps and accepts local SDK and
toolchain paths, so consumers do not need to edit the shared CMake presets.

Prerequisites include a C++23 compiler, CMake, Ninja, Rust/Cargo, LLVM/libclang,
CUDA Toolkit, TensorRT, and the repository's vcpkg checkout. The vcpkg manifest
installs OpenCV, Eigen3, GoogleTest, and nlohmann-json. The script defaults
vcpkg to `cpp/thirdparty/vcpkg` and LLVM to `C:\Program Files\LLVM\bin`;
provide different locations as parameters when needed. CUDA and TensorRT paths
can also be supplied in `CUDA_TOOLKIT_ROOT_DIR` and `TENSORRT_DIR`.

Run from the repository root in a Developer PowerShell (so MSVC tools are on
`PATH`):

```powershell
./scripts/build.ps1 `
  -CudaRoot 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2' `
  -TensorRtRoot 'C:\TensorRT-11.0.0.114' `
  -VcpkgRoot 'C:\path\to\vcpkg' `
  -LlvmBin 'C:\Program Files\LLVM\bin'
```

If CMake should use a separately installed OpenCV package instead of the
manifest-provided one, pass its CMake config directory with `-OpenCvDir`. To
build the webcam example too, add `-BuildFaceDetection`. Use
`-Configuration Debug` for a debug build; Release is the default. The script
configures CMake with the supplied paths, builds the DLL, then builds the main
Rust crate. It sets `INFERTENSORRT_BIN_DIR` so `build.rs` can find the DLL and
copies it beside the Rust output. CUDA and TensorRT runtime DLLs still need to
be discoverable when running the application, typically through `PATH`.

You can still run the steps separately: configure/build the C++ bridge with
CMake, then run `cargo build` (or `cargo build --release`). Cargo expects the
bridge under `cpp/build/x64-release` by default; set `INFERTENSORRT_BIN_DIR`
when it lives elsewhere.

## Build on Ubuntu Linux

The PowerShell helper above is for Windows. On Ubuntu, install a C++23-capable
GCC, CMake, Ninja, Rust/Cargo, and LLVM/libclang, and install CUDA Toolkit and
TensorRT for your system. The vcpkg checkout and manifest are the same as on
Windows. From the repository root, use `scripts/build-linux.sh`, supplying
local SDK paths as needed:

```bash
bash scripts/build-linux.sh \
  --cuda-root /usr/local/cuda \
  --tensorrt-root /usr/local/TensorRT \
  --vcpkg-root "$PWD/cpp/thirdparty/vcpkg" \
  --llvm-lib-dir /usr/lib/llvm-18/lib \
  --clang /usr/bin/clang
```

The script defaults CUDA, TensorRT, and vcpkg to the paths shown above.
`--llvm-lib-dir` is required unless `LIBCLANG_PATH` is set, and must contain
`libclang.so`. Use `--config Debug` for a debug build or add
`--face-detection` to build the webcam example too. To use system OpenCV instead
of the vcpkg manifest, pass `--opencv-dir /path/to/opencv/cmake`. The script
bootstraps vcpkg, builds the bridge, then builds the main Rust crate. At
runtime, CUDA and TensorRT shared libraries must be discoverable through the
loader path, commonly `LD_LIBRARY_PATH`.

To run the equivalent CMake and Cargo commands manually:

```bash
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export TENSORRT_DIR=/usr/local/TensorRT
export VCPKG_ROOT="$PWD/cpp/thirdparty/vcpkg"
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
export CLANG_PATH=/usr/bin/clang

"$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
cmake -S cpp -B cpp/build/linux-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT_DIR" \
  -DTENSORRT_DIR="$TENSORRT_DIR"
cmake --build cpp/build/linux-release
INFERTENSORRT_BIN_DIR="$PWD/cpp/build/linux-release" cargo build --release
```

Adjust the CUDA, TensorRT, and LLVM paths to match the installed locations.
`LIBCLANG_PATH` must name the directory containing `libclang.so`. For debug builds, use
`-DCMAKE_BUILD_TYPE=Debug`, build in `cpp/build/linux-debug`, and run `cargo
build` without `--release`. To use a system OpenCV install, add
`-DOpenCV_DIR=/path/to/opencv/cmake` to the CMake configure command; otherwise
vcpkg resolves OpenCV from the project manifest. At runtime, ensure CUDA and
TensorRT shared libraries are discoverable through the loader path (commonly
`LD_LIBRARY_PATH`).

## Run the Rust CLI

```powershell
cargo build
cargo run
```

On Linux, run `cargo run --release` after building with the script. For
separate Cargo builds, set `INFERTENSORRT_BIN_DIR` to the directory containing
`libinfertensorrt.so`.

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
The public header `cpp/include/public/rsinference.h` is C-compatible and exports
`rs_inference_abi_version()`. The Rust loader checks ABI version 1 before
resolving the remaining bridge functions.

TensorRT's and OpenVINO's own DLLs/shared libraries must also be discoverable
by the process.
Set `PATH` (Windows) or the platform's library search path before starting the
application. On Windows, TensorRT and CUDA runtime DLLs usually need to be on
`PATH` or beside the executable; placing them only beside the bridge DLL is
not sufficient. For OpenVINO, add `runtime\bin\intel64\Release` and
`runtime\3rdparty\tbb\bin` (Windows) or `runtime/lib/intel64` and
`runtime/3rdparty/tbb/lib` (Linux) to the search path. Match the bridge,
application, vendor runtime, and system architecture.

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
a catch-all arm. Requests for ONNX Runtime currently return
`EngineError::UnsupportedBackend`. OpenVINO is implemented, but on a machine
where its bridge was not built, `Engine::load` returns `EngineError::BridgeLoad`.

## Testing

```bash
cargo test
```

Tests that run real inference need working hardware and drivers (an NVIDIA GPU
for TensorRT, a healthy CPU runtime for OpenVINO). They are marked ignored
unless the `hardware-tests` feature is enabled:

```bash
cargo test --features hardware-tests
```

On Windows, run with `-- --test-threads=1` as CI does. Outside CI, a hardware
test whose bridge cannot be loaded prints a hint and is skipped; when the `CI`
environment variable is set, a bridge that fails to load is a hard failure.
The C++ bridge has its own GoogleTest suite, run with
`ctest --test-dir cpp/build/<preset> --output-on-failure`.

## Continuous integration

`.github/workflows/develop.yml` runs on pushes to `develop` and on pull requests
to `develop` and `master`. On Ubuntu 24.04 and Windows Server 2022 it installs
the SDKs with `scripts/install-deps.*`, builds the C++ bridge and runs `ctest`,
uploads the bridge libraries as the `bridge-linux` / `bridge-windows`
artifacts, then builds and tests the Rust crate against them. GitHub-hosted
runners have no GPU, so the hardware tests stay ignored there.

To reproduce the CI jobs locally in Docker (the working tree, including
uncommitted changes, is mounted read-only):

```bash
scripts/docker-ci.sh [all|cpp|rust] [--memory 6g] [--cpus N]
```

```powershell
./scripts/docker-ci.ps1 -Job all   # Docker Desktop in Windows-containers mode
```

Bridge libraries produced by the container land in `scripts/docker-ci/out/bridge`.

## Releases

Pushing a tag matching `v*` (for example `v0.1.0`) runs
`.github/workflows/release.yml`. It reuses the full `develop.yml` pipeline, then
builds `rsinference` in release mode on each platform and attaches one archive
per target to a GitHub Release:

| Target | Archive |
| --- | --- |
| `x86_64-unknown-linux-gnu` | `rsinference-<tag>-x86_64-unknown-linux-gnu.tar.gz` |
| `x86_64-pc-windows-msvc` | `rsinference-<tag>-x86_64-pc-windows-msvc.zip` |

Each archive contains the CLI, the prebuilt bridge libraries, and this README.
CUDA, TensorRT, and OpenVINO are not included; install them (for example with
`scripts/install-deps.*`) and make them discoverable as described in
[Native library lookup](#native-library-lookup). Keep the tag in step with the
`version` in `Cargo.toml`.

## License

MIT
