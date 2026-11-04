#!/usr/bin/env bash
# Run the Linux jobs of .github/workflows/develop.yml locally in a Linux container.
# Requires Docker (Linux containers). The working tree, including uncommitted changes,
# is mounted read-only and copied into the container. Collected bridge libraries land
# in scripts/docker-ci/out/bridge.
#
# Usage: scripts/docker-ci.sh [all|cpp|rust] [--memory 6g] [--cpus N] [--image NAME]
set -euo pipefail

job=all
image=infer-linux-ci
memory=""
cpus="$(nproc 2>/dev/null || echo 4)"
while (($#)); do
  case "$1" in
    all|cpp|rust) job="$1"; shift ;;
    --memory) memory="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ctx="$repo/scripts/docker-ci"
out="$ctx/out"
name="infer-ci-linux-$job"

docker build -t "$image" -f "$ctx/Dockerfile.linux" --build-context "scripts=$repo/scripts" "$ctx"

if [[ "$job" != rust ]]; then rm -rf "$out/bridge"; fi
mkdir -p "$out"
docker rm -f "$name" >/dev/null 2>&1 || true

mem_args=()
if [[ -n "$memory" ]]; then mem_args=(--memory "$memory"); fi

# MSYS_NO_PATHCONV keeps Git Bash on Windows from rewriting the container paths.
MSYS_NO_PATHCONV=1 docker run --rm --name "$name" --cpus "$cpus" "${mem_args[@]}" \
  -v "$repo:/src:ro" -v "$out:/out" \
  -v infer-ci-vcpkg-cache:/cache \
  "$image" /ci/run-ci.sh "$job"
