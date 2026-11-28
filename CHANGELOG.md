# Changelog

## Unreleased

### Performance

- **TensorRT no longer allocates per inference.** Each execution context
  (`n_contexts`) now owns a CUDA stream and persistent device buffers with
  pinned host staging buffers, grown only when a dynamic input gets larger.
  Inputs are read straight from the caller's `RSInferTensor` instead of being
  copied into an intermediate tensor first, and requests run with
  `cudaMemcpyAsync` + `enqueueV3` on the context's stream. On an RTX 5070 Ti
  with the UltraFace RFB-640 model, a synchronous `infer` + `get_results`
  dropped from 5.2 ms to 2.3 ms (p50); see `examples/latency.rs`.
- **`infer_async` no longer starts a thread per call.** The work is queued on a
  free context's stream and a stream callback collects the outputs.

### Behavior changes (TensorRT)

- `infer_async` validates its inputs before returning, so a wrong name, data
  type, shape or size is reported by `infer_async` itself rather than by the
  next `get_results`.
- `infer_async` waits for a free execution context when all `n_contexts` are
  busy, instead of starting an unbounded number of threads.
- Input data is matched to engine inputs by tensor name. Previously the shape
  was set by name but the data was copied by position, so inputs passed in a
  different order than the engine's were mixed up.

## 0.2.0

This release changes the C ABI. The bridge ABI version is now 2, so the Rust
crate refuses to load bridge libraries built for 0.1.x. Rebuild or re-download
the bridge together with the crate.

### Breaking changes

- **New data types.** `RSInferDataType_Int8 = 11` and `RSInferDataType_FP8E5M2 = 12`
  were added, and `RSInferDataType_Unknown` moved from 11 to 13. In Rust these
  are `Dtype::I8` and `Dtype::F8E5M2`, and `Dtype::UNKNOWN` is now 13.
- **`Dtype` is `#[non_exhaustive]`.** `match` on it needs a `_` arm.
- **8-bit signedness is checked.** Signed int8 tensors must be sent as `Dtype::I8`.
  Sending `Dtype::U8` to an int8 model input (or `I8` to a uint8 input) now fails
  on both TensorRT and OpenVINO. Use `tensor_dtype()` to find the type a model expects.
- **FP8 layouts are separate.** `Dtype::F8` / `RSInferDataType_FP8` means E4M3 only.
  OpenVINO f8e5m2 tensors are reported as `F8E5M2`. TensorRT has no E5M2 type.
- **TensorRT checks input sizes exactly.** Input data that is larger than the
  tensor's shape needs is now rejected instead of being cut off, matching OpenVINO.
- **4-bit element size is 1.** `rs_inference_runner_get_tensor_data_size` now
  returns 1 for Int4/FP4 (it returned 2), matching `Dtype::byte_size`. A buffer
  of n 4-bit elements is (n + 1) / 2 bytes; use `Dtype::byte_count` in Rust.

### Fixes

- OpenVINO and TensorRT now report Int32, Int64, Bool, UInt8, FP8 and E8M0
  tensors instead of `Unknown`, and OpenVINO models that use them now load.
- The TensorRT input data type check now runs in release builds.
