//! Live webcam face detection with the version-RFB-640 UltraFace ONNX model,
//! running through `rsinference` and OpenCV.
//!
//! Usage:
//!   facedetection [model.onnx] [camera_index] [score_threshold]
//!
//! Defaults:
//!   model           <repo>/models/version-RFB-640.onnx
//!   camera_index    0
//!   score_threshold 0.7
//!
//! The first run parses the ONNX model and builds a TensorRT engine next to it
//! (`version-RFB-640.engine`); later runs deserialize the cached engine and
//! start immediately. Press ESC or q to quit.
//!
//! At runtime the TensorRT/CUDA, OpenCV and bridge DLLs must be on PATH (the
//! bridge itself is also copied next to this executable by `build.rs`).

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::time::Instant;

use opencv::core::{self, Mat, Point, Scalar, Size, Vector, CV_32F};
use opencv::prelude::*;
use opencv::{highgui, imgproc, videoio};

use rsinference::engine::{Engine, EngineType};
use rsinference::types::{Dtype, Tensor};

const DEFAULT_MODEL: &str = "models/version-RFB-640.onnx";
const WINDOW_NAME: &str = "rsinference UltraFace (version-RFB-640)";
const NMS_IOU_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_arg = args.next();
    let camera_index = args
        .next()
        .map(|value| value.parse::<i32>())
        .transpose()?
        .unwrap_or(0);
    let score_threshold = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.7);
    if args.next().is_some() {
        eprintln!(
            "usage: {} [model.onnx] [camera_index] [score_threshold]",
            std::env::args()
                .next()
                .unwrap_or_else(|| "facedetection".into())
        );
        std::process::exit(2);
    }

    let model_path = resolve_model_path(model_arg.as_deref())?;
    println!("Model: {}", model_path.display());

    let engine = Engine::load(EngineType::TensorRT)?;
    println!("Bridge: {}", engine.engine_type());

    let instance = engine.create_infer_instance(&model_path, None)?;
    let input_name = instance.input_name(0)?;
    let input_shape = instance.input_shape(0)?;
    if input_shape.len() != 4 || input_shape[1] != 3 || input_shape.iter().any(|&dim| dim <= 0) {
        return Err(format!("unexpected model input shape: {input_shape:?}").into());
    }
    let input_height = i32::try_from(input_shape[2])?;
    let input_width = i32::try_from(input_shape[3])?;

    println!(
        "Model ready (input '{}' {}x{}x{}x{})",
        input_name, input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    );

    let mut capture = videoio::VideoCapture::new(camera_index, videoio::CAP_ANY)?;
    if !capture.is_opened()? {
        return Err(format!(
            "could not open camera {camera_index}; try another index or check camera permissions"
        )
        .into());
    }
    capture.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    capture.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;

    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)?;

    let mut fps = 0.0_f64;
    loop {
        let mut frame = Mat::default();
        if !capture.read(&mut frame)? || frame.empty() {
            break;
        }

        // Mirror the preview so it behaves like a selfie camera.
        let mut mirrored = Mat::default();
        core::flip(&frame, &mut mirrored, 1)?;

        let started = Instant::now();

        let input_data = preprocess(&mirrored, input_width, input_height)?;
        let input = Tensor {
            name: input_name.clone(),
            shape: input_shape.clone(),
            dtype: Dtype::F32,
            data: input_data,
        };

        instance.infer(&[input])?;
        let outputs = instance.get_results()?;
        let detections = postprocess(&outputs, mirrored.cols(), mirrored.rows(), score_threshold)?;

        let elapsed = started.elapsed().as_secs_f64();
        let instant_fps = if elapsed > 0.0 { 1.0 / elapsed } else { 0.0 };
        fps = if fps > 0.0 {
            0.9 * fps + 0.1 * instant_fps
        } else {
            instant_fps
        };

        draw_detections(&mut mirrored, &detections, fps)?;
        highgui::imshow(WINDOW_NAME, &mirrored)?;

        let key = highgui::wait_key(1)? & 0xFF;
        if key == 27 || key == b'q' as i32 || key == b'Q' as i32 {
            break;
        }
        if highgui::get_window_property(WINDOW_NAME, highgui::WND_PROP_VISIBLE)? < 1.0 {
            break;
        }
    }

    capture.release()?;
    highgui::destroy_all_windows()?;
    Ok(())
}

/// Resolve the default model both when the example is run from its own
/// directory and when it is run from the repository root.
fn resolve_model_path(arg: Option<&str>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(arg) = arg {
        return Ok(PathBuf::from(arg));
    }

    let candidates = [
        PathBuf::from(DEFAULT_MODEL),
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DEFAULT_MODEL),
        Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_MODEL),
    ];

    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            format!(
                "model not found; pass it as the first argument or place it at {}",
                DEFAULT_MODEL
            )
            .into()
        })
}

/// Resize to the model input, convert BGR -> RGB, normalize to
/// `(x - 127) / 128` and lay the image out as planar float32 CHW.
fn preprocess(frame: &Mat, width: i32, height: i32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color_def(frame, &mut rgb, imgproc::COLOR_BGR2RGB)?;

    let mut resized = Mat::default();
    imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut normalized = Mat::default();
    resized.convert_to(&mut normalized, CV_32F, 1.0 / 128.0, -127.0 / 128.0)?;

    let mut channels = Vector::<Mat>::new();
    core::split(&normalized, &mut channels)?;

    let plane_bytes = width as usize * height as usize * std::mem::size_of::<f32>();
    let mut bytes = Vec::with_capacity(plane_bytes * 3);
    for index in 0..channels.len() {
        let channel = channels.get(index)?;
        let data = channel.data_bytes()?;
        if data.len() != plane_bytes {
            return Err(format!(
                "unexpected channel size: got {} bytes, expected {plane_bytes}",
                data.len()
            )
            .into());
        }
        bytes.extend_from_slice(data);
    }

    Ok(bytes)
}

/// The ONNX graph already applies softmax to `scores` and decodes `boxes`, so
/// only thresholding, scaling to the frame and NMS are left to do here.
fn postprocess(
    outputs: &[Tensor],
    frame_width: i32,
    frame_height: i32,
    score_threshold: f32,
) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
    let scores = outputs
        .iter()
        .find(|tensor| tensor.name == "scores")
        .or_else(|| outputs.first())
        .ok_or("model returned no scores tensor")?;
    let boxes = outputs
        .iter()
        .find(|tensor| tensor.name == "boxes")
        .or_else(|| outputs.get(1))
        .ok_or("model returned no boxes tensor")?;

    let score_data = as_f32_slice(scores)?;
    let box_data = as_f32_slice(boxes)?;

    let score_stride = usize::try_from(scores.shape.last().copied().unwrap_or(2))
        .map_err(|_| "invalid scores tensor shape")?;
    let num_anchors = if scores.shape.len() >= 3 {
        usize::try_from(scores.shape[1]).map_err(|_| "invalid scores tensor shape")?
    } else {
        score_data.len() / score_stride.max(1)
    };
    let box_stride = usize::try_from(boxes.shape.last().copied().unwrap_or(4))
        .map_err(|_| "invalid boxes tensor shape")?;

    if score_stride < 2
        || box_stride < 4
        || score_data.len() < num_anchors * score_stride
        || box_data.len() < num_anchors * box_stride
    {
        return Err(format!(
            "unexpected output tensor layout (scores {:?}, boxes {:?})",
            scores.shape, boxes.shape
        )
        .into());
    }

    let max_x = frame_width as f32 - 1.0;
    let max_y = frame_height as f32 - 1.0;

    let mut detections = Vec::new();
    for anchor in 0..num_anchors {
        // Class 0 is background; class 1 is face.
        let score = score_data[anchor * score_stride + 1];
        if score < score_threshold {
            continue;
        }

        let offset = anchor * box_stride;
        let x1 = (box_data[offset] * frame_width as f32).clamp(0.0, max_x);
        let y1 = (box_data[offset + 1] * frame_height as f32).clamp(0.0, max_y);
        let x2 = (box_data[offset + 2] * frame_width as f32).clamp(0.0, max_x);
        let y2 = (box_data[offset + 3] * frame_height as f32).clamp(0.0, max_y);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            score,
        });
    }

    Ok(apply_nms(detections, NMS_IOU_THRESHOLD))
}

fn as_f32_slice(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if tensor.dtype != Dtype::F32 {
        return Err(format!(
            "tensor '{}' has dtype {:?}, expected F32",
            tensor.name, tensor.dtype
        )
        .into());
    }
    if tensor.data.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!(
            "tensor '{}' payload is not a whole number of f32 values",
            tensor.name
        )
        .into());
    }

    Ok(tensor
        .data
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn intersection_over_union(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let overlap_width = x2 - x1;
    let overlap_height = y2 - y1;
    if overlap_width <= 0.0 || overlap_height <= 0.0 {
        return 0.0;
    }

    let intersection = overlap_width * overlap_height;
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    intersection / (area_a + area_b - intersection)
}

fn apply_nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let mut kept = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(detections[i]);
        for j in i + 1..detections.len() {
            if !suppressed[j]
                && intersection_over_union(&detections[i], &detections[j]) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }
    kept
}

fn draw_detections(frame: &mut Mat, detections: &[Detection], fps: f64) -> opencv::Result<()> {
    let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
    for detection in detections {
        let rect = core::Rect::new(
            detection.x1 as i32,
            detection.y1 as i32,
            (detection.x2 - detection.x1).max(1.0) as i32,
            (detection.y2 - detection.y1).max(1.0) as i32,
        );
        imgproc::rectangle(frame, rect, green, 2, imgproc::LINE_AA, 0)?;

        let label = format!("{:.2}", detection.score);
        let origin = Point::new(rect.x, (rect.y - 6).max(12));
        imgproc::put_text(
            frame,
            &label,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            green,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }

    let status = format!("FPS {fps:.1} | faces {}", detections.len());
    imgproc::put_text(
        frame,
        &status,
        Point::new(10, 28),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar::new(0.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}
