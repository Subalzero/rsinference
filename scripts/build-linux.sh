#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/build-linux.sh [options]

Build the C++ TensorRT bridge and Rust crate on Linux.

Options:
  --cuda-root PATH       CUDA Toolkit root (default: CUDA_TOOLKIT_ROOT_DIR or /usr/local/cuda)
  --tensorrt-root PATH   TensorRT root (default: TENSORRT_DIR or /usr/local/TensorRT)
  --vcpkg-root PATH      vcpkg checkout (default: VCPKG_ROOT or cpp/thirdparty/vcpkg)
  --llvm-lib-dir PATH    Directory containing libclang.so (default: LIBCLANG_PATH)
  --clang PATH           clang executable (default: CLANG_PATH or clang)
  --opencv-dir PATH      OpenCV CMake config directory (optional)
  --config NAME          Debug or Release (default: Release)
  --face-detection       Also build examples/facedetection
  -h, --help             Show this help
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_root="${CUDA_TOOLKIT_ROOT_DIR:-/usr/local/cuda}"
tensorrt_root="${TENSORRT_DIR:-/usr/local/TensorRT}"
vcpkg_root="${VCPKG_ROOT:-$repo_root/cpp/thirdparty/vcpkg}"
llvm_lib_dir="${LIBCLANG_PATH:-}"
clang_bin="${CLANG_PATH:-clang}"
opencv_dir=""
configuration="Release"
build_face_detection=0

while (($#)); do
  case "$1" in
    --cuda-root|--tensorrt-root|--vcpkg-root|--llvm-lib-dir|--clang|--opencv-dir|--config)
      if (($# < 2)); then echo "Missing value for $1" >&2; usage >&2; exit 2; fi
      option="$1"; value="$2"; shift 2
      case "$option" in
        --cuda-root) cuda_root="$value" ;;
        --tensorrt-root) tensorrt_root="$value" ;;
        --vcpkg-root) vcpkg_root="$value" ;;
        --llvm-lib-dir) llvm_lib_dir="$value" ;;
        --clang) clang_bin="$value" ;;
        --opencv-dir) opencv_dir="$value" ;;
        --config) configuration="$value" ;;
      esac
      ;;
    --face-detection) build_face_detection=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$configuration" != Debug && "$configuration" != Release ]]; then
  echo "Configuration must be Debug or Release (got '$configuration')." >&2
  exit 2
fi
if [[ -z "$llvm_lib_dir" ]]; then
  echo "Set LIBCLANG_PATH or pass --llvm-lib-dir to the directory containing libclang.so." >&2
  exit 2
fi

for path in "$cuda_root/include/cuda.h" \
            "$tensorrt_root/include/NvInferVersion.h" \
            "$vcpkg_root/scripts/buildsystems/vcpkg.cmake" \
            "$llvm_lib_dir/libclang.so"; do
  if [[ ! -e "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done
if ! command -v "$clang_bin" >/dev/null 2>&1 && [[ ! -x "$clang_bin" ]]; then
  echo "clang executable not found: $clang_bin" >&2
  exit 1
fi

build_dir="$repo_root/cpp/build/linux-${configuration,,}"
toolchain_file="$vcpkg_root/scripts/buildsystems/vcpkg.cmake"

cmake_args=(
  -S "$repo_root/cpp"
  -B "$build_dir"
  -G Ninja
  "-DCMAKE_BUILD_TYPE=$configuration"
  -DCMAKE_C_COMPILER=gcc
  -DCMAKE_CXX_COMPILER=g++
  "-DCMAKE_TOOLCHAIN_FILE=$toolchain_file"
  "-DCUDA_TOOLKIT_ROOT_DIR=$cuda_root"
  "-DCUDAToolkit_ROOT=$cuda_root"
  "-DTENSORRT_DIR=$tensorrt_root"
)
if [[ -n "$opencv_dir" ]]; then cmake_args+=("-DOpenCV_DIR=$opencv_dir"); fi

"$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config "$configuration"

export CLANG_PATH="$clang_bin"
export LIBCLANG_PATH="$llvm_lib_dir"
export INFERTENSORRT_BIN_DIR="$build_dir"
if [[ "$configuration" == Release ]]; then
  cargo build --manifest-path "$repo_root/Cargo.toml" --release
else
  cargo build --manifest-path "$repo_root/Cargo.toml"
fi

if ((build_face_detection)); then
  vcpkg_installed="$build_dir/vcpkg_installed/x64-linux"
  export OPENCV_LINK_PATHS="$vcpkg_installed/lib"
  export OPENCV_INCLUDE_PATHS="$vcpkg_installed/include/opencv4"
  export OPENCV_LINK_LIBS='opencv_core,opencv_imgproc,opencv_videoio,opencv_highgui'
  if [[ "$configuration" == Release ]]; then
    cargo build --manifest-path "$repo_root/examples/facedetection/Cargo.toml" --release
  else
    cargo build --manifest-path "$repo_root/examples/facedetection/Cargo.toml"
  fi
fi
