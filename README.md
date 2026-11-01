# RSInference

A Rust inference bridge with **pluggable native backends**.

The Rust side owns the safe API; the actual inference work happens in a small C++
bridge DLL per backend. Bridges are **loaded at runtime and never linked at build
time**, so a Rust binary does not need a vendor SDK at link time, and one binary
can talk to several backends.

The bridge is **not optional**: this crate always compiles the native plumbing,
so `libclang` and `cpp/include/public/rsinference.h` are required to build it.
Only the vendor runtimes (TensorRT/CUDA, ...) are installed separately, by you.

```text
  your Rust code
        │
        ▼
  rsinference (Engine, EngineType, Bridge)
        │   libloading -- runtime symbol resolution
        ▼
  infer<backend>_cpp.dll / .so / .dylib
        │
        ▼
  backend runtime  (TensorRT, ONNX Runtime, OpenVINO, ...)
        │
        ▼
  GPU / CPU
```

> **Status:** the TensorRT backend is implemented. The ONNX Runtime and OpenVINO
> backends are declared in the Rust API (`EngineType`) but their C++ bridges do
> not exist yet, and the inference call itself is not wired up — see
> [Status](#status).

## Backends

| `EngineType` | Bridge library | C++ bridge | Status |
|---|---|---|---|
| `EngineType::TensorRT` | `infertensorrt_cpp` | `cpp/` | implemented |
| `EngineType::ONNX` | `inferonnx_cpp` | — | planned |
| `EngineType::OpenVINO` | `inferopenvino_cpp` | — | planned |

The bridge file name is resolved per platform: `infertensorrt_cpp.dll`,
`libinfertensorrt_cpp.so`, `libinfertensorrt_cpp.dylib`.

## Adding it to your crate

```toml
[dependencies]
rsinference = { path = "../rsinference" }
```

## Quick start

```powershell
# building requires libclang + cpp/include/public/rsinference.h
cargo build

# run the CLI (the bridge library must be resolvable at runtime)
cargo run -- path\to\model.engine
```

## Library usage

```rust
use rsinference::{Engine, EngineType};

fn main() -> Result<(), rsinference::InferError> {
    let engine = Engine::load("model.engine", EngineType::TensorRT)?;
    println!("bridge: {}", engine.bridge().path.display());
    Ok(())
}
```

`Engine::load` opens the model file, then loads the bridge for the requested
backend:

```rust
Engine::load("model.engine", EngineType::TensorRT)?;
```

The model path is used exactly as given. The bridge library is resolved the same
way as `Bridge::load_default_backend()`: first next to the executable, then by
its bare file name through the platform search order (`PATH`).

If you prefer to find the library yourself, use `Bridge` directly:

```rust
use rsinference::{Bridge, EngineType};

let bridge = Bridge::load_default_backend()?;              // exe dir -> PATH
let bridge = Bridge::load("C:/sdk/my_bridge.dll")?;         // explicit file
let bridge = Bridge::load_backend("inferonnx_cpp")?;        // by backend name
```

`Bridge::load_default_backend()` searches, in order:

1. the bridge library next to the current executable,
2. the bare library file name, resolved through the platform search order (`PATH`).

Keep the `Bridge` alive for as long as you use anything it produced: dropping it
unloads the library and invalidates every function pointer it handed out.

`Bridge` and `Engine` are deliberately **not** `Send`/`Sync`: native backend
runtimes are not thread-safe, so one instance belongs to one thread.

## Native requirements

Bridges ship as separate native libraries and are **not** bundled with this
crate. Their runtime dependencies are **not** bundled either.

### You must install and locate these yourself

| Component | Needed by | Note |
|---|---|---|
| GPU driver | TensorRT | must match your GPU |
| CUDA Toolkit runtime | TensorRT | version must match the TensorRT build |
| TensorRT runtime | TensorRT | `nvinfer_*.dll` etc. |
| ONNX Runtime | ONNX | planned |
| OpenVINO runtime | OpenVINO | planned |
| the bridge library itself | all | built from this repo, see below |

Installing these is the responsibility of the user or the consuming
application. This crate does not download, vendor, or install vendor SDKs.

### Making the vendor libraries resolvable

Put the bridge library where `Engine::load` / `Bridge::load_default_backend()`
looks (see above), **and** make the backend's own libraries resolvable — usually by
adding their directories to `PATH` before starting the program:

```powershell
$env:PATH = "C:\TensorRT-10.8\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
.\target\release\my-app.exe
```

`PATH` must be set **before** the process starts. Changing it after launch does
not reliably affect the loader.

> **Important:** putting `nvinfer_*.dll` next to the bridge library is **not**
> enough on Windows. When a DLL is loaded, its own dependencies are resolved
> from the *executable's* directory and `PATH`, not from the loaded DLL's
> directory. Either add the vendor `bin`/`lib` directory to `PATH` or copy the
> vendor DLLs next to your `.exe`.

In-process alternatives to `PATH` are `SetDllDirectoryW` / `AddDllDirectory`
(before loading the bridge). Architecture must match: 64-bit bridge for a
64-bit build, plus the matching MSVC runtime.

### Build requirements

These are required for **every** build of this crate — the native bridge is not
feature-gated.

| Requirement | Why |
|---|---|
| `libclang` / LLVM (with its `bin` on `PATH`) | `bindgen` generates the FFI types |
| `cpp/include/public/rsinference.h` | parsed by `build.rs`; missing header = build error |
| `INFERTENSORRT_BIN_DIR` | optional, points the build at a custom DLL directory |

`build.rs` copies the bridge next to the built binary so
`Bridge::load_default_backend()` finds it through the executable directory.
Nothing machine-specific is baked into the crate.

## Errors

```rust
#[non_exhaustive]
pub enum InferError {
    EngineLoad { path: String, source: std::io::Error },   // model missing/unreadable/not a file
    BridgeLoad { path: String, message: String },          // bridge library or a symbol missing
    Backend { message: String },                           // backend reported a failure
    ShapeMismatch { expected: String, actual: String },    // input length vs engine input
    UnsupportedBackend { engine_type: EngineType },        // no bridge for this backend yet
}
```

`InferError` is `#[non_exhaustive]`, so downstream `match` arms need a `_`
catch-all.

- `BridgeLoad` with `file not found` → the bridge library is not where the loader
  looked (see [Native requirements](#native-requirements)).
- `BridgeLoad` with a missing symbol → the bridge is an older/newer build than
  the Rust side expects.
- `Backend` → the bridge loaded fine and the backend itself failed (typically a
  missing GPU, driver, or vendor runtime library).
- `UnsupportedBackend` → the backend has no C++ bridge in this repository yet
  (today: `OpenVINO`, `ONNX`).

## Status

- [x] Runtime-loaded bridge, no link-time dependency
- [x] TensorRT C++ bridge and FFI surface
- [x] `Engine` / `EngineType` / `Bridge` API
- [x] Native bridge is mandatory (no feature gate, always compiled)
- [x] Unimplemented backends rejected with `UnsupportedBackend`
- [x] `Bridge`/`Engine` are `!Send`/`!Sync` (native runtimes are not thread-safe)
- [ ] `Engine::infer` native path
- [ ] Engine input/output shape discovery from the backend
- [ ] ONNX Runtime bridge (`inferonnx_cpp`)
- [ ] OpenVINO bridge (`inferopenvino_cpp`)
- [ ] Per-backend runtime dependency checks

## License

MIT
