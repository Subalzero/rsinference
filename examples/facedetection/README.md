# Face detection example

Live webcam face detection with the
[version-RFB-640 UltraFace model](../../models/README.md), running through
[`rsinference`](../../README.md) and OpenCV.

It is the Rust counterpart of
[`cpp/src/samples/tensorrtfacedetecttest.cpp`](../../cpp/src/samples/tensorrtfacedetecttest.cpp).

## What it does

1. Opens the webcam with OpenCV and mirrors the preview.
2. Resizes the frame to the model input (640×480), converts BGR → RGB,
   normalizes with `(x - 127) / 128` and lays it out as planar `float32`
   NCHW.
3. Runs inference through `rsinference` (`Engine` + `Tensor`).
4. Thresholds the `scores` output, decodes `boxes`, applies hard NMS and draws
   the surviving faces.
5. Shows the preview with FPS and face count. Press `ESC` or `q` to quit.

The first run builds a TensorRT engine next to the ONNX file
(`models/version-RFB-640.engine`); later runs reuse it.

## Requirements

- OpenCV 4 (the `cpp` vcpkg manifest already installs `opencv4`)
- CUDA + TensorRT, with the runtime `bin`/`lib` directories on `PATH`
- The `infertensorrt` bridge DLL, built from `cpp/`
- `libclang` on `PATH` while building

## Build

Build the C++ bridge first, in **Release** mode (a Debug bridge needs the debug
MSVC runtime and is noticeably slower). From a Developer PowerShell:

```powershell
$repo = Resolve-Path ..\..
cmake -S "$repo\cpp" -B "$repo\cpp\build\x64-release" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE="$repo\cpp\thirdparty\vcpkg\scripts\buildsystems\vcpkg.cmake" `
    -DVCPKG_TARGET_TRIPLET=x64-windows `
    -DVCPKG_MANIFEST_DIR="$repo\cpp" `
    -DTENSORRT_DIR=C:\TensorRT-11.0.0.114
cmake --build "$repo\cpp\build\x64-release"
```

`build.rs` also looks for the bridge in that build directory, so an install
step is not required.

If OpenCV was installed with the repo's vcpkg manifest, point the `opencv`
crate at that install (paths are for the default vcpkg output tree):

```powershell
$vcpkg = Resolve-Path ..\..\cpp\build\x64-release\vcpkg_installed\x64-windows
$env:OPENCV_LINK_PATHS    = "$vcpkg\lib"
$env:OPENCV_LINK_LIBS     = "opencv_core4,opencv_imgproc4,opencv_videoio4,opencv_highgui4"
$env:OPENCV_INCLUDE_PATHS = "$vcpkg\include\opencv4"

cargo build
```

If OpenCV is installed system-wide, the crate's normal OpenCV discovery can be
used instead (for example `OpenCV_DIR`, or vcpkg via `VCPKG_ROOT` and
`VCPKGRS_DYNAMIC=1`).

## Run

Add the runtime directories to `PATH` before starting the process. On Windows,
for example:

```powershell
$env:PATH = "C:\TensorRT-11.0.0.114\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;$vcpkg\bin;$env:PATH"

cargo run
```

The bridge DLL is copied next to the example's executable by the crate's
`build.rs`, so it does not need to be added to `PATH` separately.

### Arguments

```text
facedetection [model.onnx] [camera_index] [score_threshold]
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `model.onnx` | `../../models/version-RFB-640.onnx` | ONNX model path |
| `camera_index` | `0` | OpenCV camera index |
| `score_threshold` | `0.7` | Minimum face confidence |

Examples:

```powershell
cargo run
cargo run -- ..\..\models\version-RFB-640.onnx 1 0.6
```
