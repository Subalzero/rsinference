# version-RFB-640

Ultra-lightweight face detection model in ONNX format, exported from
[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
(RFB variant, 640×480 input). It is included in this repository as the sample
model for the TensorRT runner.

## File

| | |
| --- | --- |
| File | `version-RFB-640.onnx` |
| Size | 1,588,012 bytes (1.51 MiB) |
| Task | Face detection (bounding boxes) |
| Architecture | Ultra-Light Fast Generic Face Detector 1MB, RFB variant |
| ONNX IR version | 4 |
| Opset | 9 |
| Producer | `pytorch 1.3` |
| Graph | 242 nodes, opset-9 operators only (Conv, BatchNormalization, Relu, Concat, Reshape, Transpose, Softmax, …) |
| SHA-256 | `8f4c659275977e7a3bfbfa339a9c769ad793df50f9c0baa8c14b11baa1646430` |

The checksum matches the upstream file on
[Hugging Face](https://huggingface.co/onnxmodelzoo/version-RFB-640), so this is
an unmodified copy.

## Description

A single-stage, anchor-based face detector intended for edge devices. The
backbone is a depthwise-separable MobileNet-style network with RFB
(receptive-field block) feature fusion, followed by four detection heads at
strides 8, 16, 32 and 64. The model is small enough for CPU and embedded GPU
inference.

- **Training data:** WIDER FACE, using the cleaned RetinaFace annotations.
  See [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641).
- **Upstream source:** [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- **ONNX Model Zoo card:** [onnxmodelzoo/version-RFB-640](https://huggingface.co/onnxmodelzoo/version-RFB-640)

## Inputs and outputs

The model has **fixed shapes** (batch 1). It is not a dynamic-shape export; to
run other resolutions, re-export the PyTorch model with dynamic axes.

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input` | `float32` | `1 × 3 × 480 × 640` | RGB image, NCHW, already normalized |
| `scores` | `float32` | `1 × 17640 × 2` | Softmax probabilities per anchor: `[background, face]` |
| `boxes` | `float32` | `1 × 17640 × 4` | Decoded boxes in corner form `[x1, y1, x2, y2]`, in normalized image coordinates |

`17640` is the total anchor count for a 640×480 input:

| Level | Feature map (W × H) | Anchors/cell | Anchors |
| --- | --- | --- | --- |
| 1 (stride 8) | 80 × 60 | 3 | 14,400 |
| 2 (stride 16) | 40 × 30 | 2 | 2,400 |
| 3 (stride 32) | 20 × 15 | 2 | 600 |
| 4 (stride 64) | 10 × 8 | 3 | 240 |
| | | | **17,640** |

Both post-processing steps normally done by the detector are **already part of
the graph**:

- a `Softmax` produces `scores` directly (no extra softmax needed), and
- prior decoding is baked in, so `boxes` are corner-form coordinates relative
  to the input image, ready to be multiplied by the original image size.

## Preprocessing

Normalization is **not** part of the graph — the first node is a `Conv` reading
`input` directly. Apply it yourself. The upstream recipe (OpenCV, BGR input):

```python
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 480))              # width, height
image = (image.astype(np.float32) - 127.0) / 128.0
image = np.transpose(image, (2, 0, 1))             # HWC -> CHW
image = np.expand_dims(image, axis=0)              # NCHW, batch = 1
```

Summary: RGB, resize to **640×480**, mean `127`, scale `1/128`, NCHW
`float32`.

## Postprocessing

1. Take the face score per anchor, `scores[0, :, 1]` (index 0 is background).
2. Drop anchors below a confidence threshold (the upstream demo uses `0.7`).
3. Apply hard non-maximum suppression (`IoU` threshold `0.5`).
4. Scale the surviving boxes by the **original** image width and height:

```python
boxes[:, [0, 2]] *= orig_width
boxes[:, [1, 3]] *= orig_height
```

Reference implementation:
[`dependencies/box_utils.py`](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/ultraface/dependencies/box_utils.py)
(`predict()` / `hard_nms()`).

## Using with this project

The C++ runner in `cpp/` loads this file through `inferrs::RunnerParams`:

```cpp
inferrs::RunnerParams params;
params.onnxFileName = "version-RFB-640.onnx";
params.dataDirs = { "models" }; // locateFile() needs at least one entry
params.nContexts = 1;

inferrs::TensorRTRunner runner(params);
runner.build();
```

Run this from the repository root. `RunnerParams::onnxFileName` is resolved
with `samplesCommon::locateFile()`, which prefixes `onnxFileName` with each
`dataDirs` entry — so the file above resolves to `models/version-RFB-640.onnx`.

Note that the serialized-engine cache path is derived from `onnxFileName` alone
(not `dataDirs`), so the build writes `version-RFB-640.engine` into the current
working directory and reuses it on the next run.

Feed an `inferrs::Tensor` named `input` with shape `{1, 3, 480, 640}`,
`dtype = inferrs::DataType::FP32` and the preprocessed bytes above. The output
tensors are named `scores` and `boxes`.

Because the ONNX graph is fixed-shape, the runner's dynamic-shape path is not
needed; `RunnerParams::fp16` / `bf16` / `int8` have no effect on TensorRT 11
(strong typing, precision follows the graph). Convert the ONNX model to FP16
before parsing if you need half precision.

## License

The upstream model and source code are released under the
**MIT License** (see the model card and the
[Ultra-Light repository](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)).
Note that the Hugging Face repository metadata tags the model as `apache-2.0`;
the model card body and upstream project both state MIT.

## References

- [ONNX Model Zoo — UltraFace](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
- [Hugging Face — onnxmodelzoo/version-RFB-640](https://huggingface.co/onnxmodelzoo/version-RFB-640)
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- [RetinaFace (arXiv:1905.00641)](https://arxiv.org/abs/1905.00641)
