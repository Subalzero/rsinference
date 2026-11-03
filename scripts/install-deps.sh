#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install-deps.sh [options] [component...]

Download and install the native SDKs needed to build the C++ bridge on
Ubuntu 24.04 (x86_64). Components: cuda, tensorrt, openvino, toolchain.
With no component, everything is installed.

  cuda        CUDA Toolkit (cudart + nvcc) from the NVIDIA apt repository
  tensorrt    TensorRT (runtime, plugins, ONNX parser) from the NVIDIA apt repository
  openvino    OpenVINO runtime archive from storage.openvinotoolkit.org
  toolchain   gcc/g++ 14, cmake, ninja, libclang (for bindgen), vcpkg prerequisites

Options:
  --deps-dir PATH   Where OpenVINO and the TensorRT root layout are placed
                    (default: DEPS_DIR or $HOME/deps)
  -h, --help        Show this help

Versions can be overridden through the environment:
  CUDA_VERSION       default 13-2 (apt package suffix)
  TENSORRT_VERSION   default 11.0.0.114-1+cuda13.2 (apt package version)
  OPENVINO_VERSION   default 2026.4.1
  OPENVINO_BUILD     default 22982.07f9c262b05
  UBUNTU_REPO        default ubuntu2404

After installing, the script prints (and appends to $GITHUB_ENV when set) the
variables the build expects: CUDA_TOOLKIT_ROOT_DIR, TENSORRT_DIR, OPENVINO_DIR,
LIBCLANG_PATH. Uses sudo for apt when not running as root.
USAGE
}

cuda_version="${CUDA_VERSION:-13-2}"
tensorrt_version="${TENSORRT_VERSION:-11.0.0.114-1+cuda13.2}"
openvino_version="${OPENVINO_VERSION:-2026.4.1}"
openvino_build="${OPENVINO_BUILD:-22982.07f9c262b05}"
ubuntu_repo="${UBUNTU_REPO:-ubuntu2404}"
deps_dir="${DEPS_DIR:-$HOME/deps}"
components=()

while (($#)); do
  case "$1" in
    --deps-dir)
      if (($# < 2)); then echo "Missing value for $1" >&2; usage >&2; exit 2; fi
      deps_dir="$2"; shift 2 ;;
    cuda|tensorrt|openvino|toolchain) components+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option or component: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if ((${#components[@]} == 0)); then components=(toolchain cuda tensorrt openvino); fi

sudo_cmd=()
if [[ "$(id -u)" -ne 0 ]]; then sudo_cmd=(sudo); fi

want() {
  local c
  for c in "${components[@]}"; do [[ "$c" == "$1" ]] && return 0; done
  return 1
}

export_var() {
  echo "export $1=$2"
  if [[ -n "${GITHUB_ENV:-}" ]]; then echo "$1=$2" >> "$GITHUB_ENV"; fi
}

apt_install() { "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"; }

nvidia_repo_ready=0
add_nvidia_repo() {
  if ((nvidia_repo_ready)); then return; fi
  local keyring
  keyring="$(mktemp -d)/cuda-keyring.deb"
  curl -fsSL -o "$keyring" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  "${sudo_cmd[@]}" dpkg -i "$keyring"
  "${sudo_cmd[@]}" apt-get update
  nvidia_repo_ready=1
}

mkdir -p "$deps_dir"
deps_dir="$(cd "$deps_dir" && pwd)"

if want toolchain; then
  "${sudo_cmd[@]}" apt-get update
  apt_install build-essential gcc-14 g++-14 cmake ninja-build curl git pkg-config \
    zip unzip tar ca-certificates libclang-dev clang \n    make autoconf autoconf-archive automake libtool m4 perl python3 bison flex \n    linux-libc-dev
  # Newest libclang the distro ships; bindgen loads libclang.so from here.
  libclang_dir="$(dirname "$(ls /usr/lib/llvm-*/lib/libclang*.so* | sort -V | tail -n1)")"
  export_var LIBCLANG_PATH "$libclang_dir"
fi

if want cuda; then
  add_nvidia_repo
  apt_install "cuda-cudart-dev-${cuda_version}" "cuda-nvcc-${cuda_version}"
  export_var CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${cuda_version//-/.}"
fi

if want tensorrt; then
  add_nvidia_repo
  trt_major="${tensorrt_version%%.*}"
  apt_install \
    "libnvinfer${trt_major}=${tensorrt_version}" \
    "libnvinfer-headers-dev=${tensorrt_version}" \
    "libnvinfer-safe-headers-dev=${tensorrt_version}" \
    "libnvinfer-headers-plugin-dev=${tensorrt_version}" \
    "libnvinfer-dev=${tensorrt_version}" \
    "libnvinfer-plugin${trt_major}=${tensorrt_version}" \
    "libnvinfer-plugin-dev=${tensorrt_version}" \
    "libnvonnxparsers${trt_major}=${tensorrt_version}" \
    "libnvonnxparsers-dev=${tensorrt_version}"
  # CMake expects a single root with include/ and lib/; the apt packages scatter
  # them across the multiarch directories, so expose them through one root.
  multiarch="$(gcc -print-multiarch)"
  trt_root="$deps_dir/TensorRT"
  mkdir -p "$trt_root"
  ln -sfn "/usr/include/$multiarch" "$trt_root/include"
  ln -sfn "/usr/lib/$multiarch" "$trt_root/lib"
  if [[ ! -f "$trt_root/include/NvInferVersion.h" ]]; then
    echo "TensorRT headers not found under $trt_root/include" >&2
    exit 1
  fi
  export_var TENSORRT_DIR "$trt_root"
fi

if want openvino; then
  archive="openvino_toolkit_ubuntu24_${openvino_version}.${openvino_build}_x86_64"
  url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${openvino_version}/linux/${archive}.tgz"
  ov_root="$deps_dir/openvino"
  if [[ ! -d "$deps_dir/$archive" ]]; then
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/openvino.tgz" "$url"
    tar -xzf "$tmp/openvino.tgz" -C "$deps_dir"
    rm -rf "$tmp"
  fi
  ln -sfn "$deps_dir/$archive" "$ov_root"
  "${sudo_cmd[@]}" "$ov_root/install_dependencies/install_openvino_dependencies.sh" -y
  # CMake looks for ${OPENVINO_DIR}/cmake, which lives under runtime/.
  export_var OPENVINO_DIR "$ov_root/runtime"
fi
