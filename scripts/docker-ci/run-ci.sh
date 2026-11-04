#!/usr/bin/env bash
# Local emulation of the cpp and rust jobs in .github/workflows/develop.yml.
set -euo pipefail
job="${1:-all}"
case "$job" in all|cpp|rust) ;; *) echo "Usage: run-ci.sh [all|cpp|rust]" >&2; exit 2 ;; esac

step() {
  echo; echo "##### STEP: $1 #####"
  local start=$SECONDS
  shift
  "$@"
  echo "##### OK ($((SECONDS - start))s) #####"
}

checkout() {
  rsync -a --delete \
    --exclude=/target --exclude=/cpp/build --exclude=/cpp/vcpkg_installed --exclude=/cpp/install \
    --exclude=/.vcpkg-cache --exclude=/bridge --exclude=/scripts/docker-ci/out \
    --exclude=/cpp/thirdparty/vcpkg/buildtrees --exclude=/cpp/thirdparty/vcpkg/downloads \
    --exclude=/cpp/thirdparty/vcpkg/packages --exclude=/cpp/thirdparty/vcpkg/installed \
    --exclude=/cpp/thirdparty/vcpkg/vcpkg \
    /src/ /work/
  # The Windows checkout may carry CRLF line endings, which break bash scripts.
  find /work/scripts /work/cpp/thirdparty/vcpkg -maxdepth 2 -name '*.sh' -exec sed -i 's/\r$//' {} +
  git config --global --add safe.directory '*'
}

dump_vcpkg_logs() {
  for f in /work/cpp/thirdparty/vcpkg/buildtrees/*/config-*-out.log /work/cpp/thirdparty/vcpkg/buildtrees/*/config-*-meson-log.txt.log; do
    [[ -f "$f" ]] && { echo "=== $f"; tail -40 "$f"; }
  done
}
trap 'rc=$?; ((rc)) && dump_vcpkg_logs; exit $rc' EXIT

ws=/work
cd /
step 'checkout (copy working tree)' checkout
cd "$ws"

export CARGO_TERM_COLOR=always

# Variables the install-deps step exported (CUDA_TOOLKIT_ROOT_DIR, TENSORRT_DIR, LIBCLANG_PATH, ...).
set -a; . <(sed "s/^/export /" /ci/github_env); set +a
# Same loader path the workflow's test steps use.
export LD_LIBRARY_PATH="$TENSORRT_DIR/lib:$OPENVINO_DIR/lib/intel64:$OPENVINO_DIR/3rdparty/tbb/lib:$CUDA_TOOLKIT_ROOT_DIR/lib64:${LD_LIBRARY_PATH:-}"

if [[ "$job" == all || "$job" == cpp ]]; then
  export VCPKG_ROOT="$ws/cpp/thirdparty/vcpkg"
  export VCPKG_DEFAULT_BINARY_CACHE=/cache/vcpkg-binary
  export VCPKG_DOWNLOADS=/cache/vcpkg-downloads
  export BUILD_DIR=cpp/build/linux-release

  step 'Bootstrap vcpkg' bash -c 'mkdir -p "$VCPKG_DEFAULT_BINARY_CACHE" "$VCPKG_DOWNLOADS" && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics'

  step 'Configure' cmake -S cpp -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_TOOLCHAIN_FILE="$ws/cpp/thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR" \
    -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT_DIR" \
    -DTENSORRT_DIR="$TENSORRT_DIR" \
    -DOPENVINO_DIR="$OPENVINO_DIR" \
    -DBUILD_EXAMPLES=OFF

  step 'Build' cmake --build "$BUILD_DIR"

  test_step() {
    ctest --test-dir "$BUILD_DIR" --output-on-failure
  }
  step 'Test' test_step

  collect() {
    mkdir -p bridge /out/bridge
    cp -L "$BUILD_DIR"/libinfer*.so bridge/
    ls -l bridge
    cp -L bridge/*.so /out/bridge/
  }
  step 'Collect bridge libraries' collect
fi

if [[ "$job" == all || "$job" == rust ]]; then
  fetch_bridge() { mkdir -p bridge; cp -L /out/bridge/*.so bridge/; ls -l bridge; }
  step 'download-artifact bridge-linux' fetch_bridge
  export INFERTENSORRT_BIN_DIR="$ws/bridge"
  step 'Build' cargo build --verbose
  step 'Test' cargo test --verbose
fi
echo; echo "##### JOB $job SUCCEEDED #####"
