# Changelog

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
