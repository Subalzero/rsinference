//! Latency benchmark for an rsinference backend.
//!
//! Usage:
//!   cargo run --release --example latency -- [model] [iterations] [warmup] [backend] [async_depth] [json_config]
//!
//! Defaults: models/version-RFB-640.onnx, 1000 iterations, 100 warm-up runs,
//! backend `tensorrt` (or `openvino`), async depth 4, no backend config.
//! For example, `'{"n_contexts":4}'` lets TensorRT run 4 async requests at once.
//!
//! It reports wall-clock latency per call, as the caller sees it:
//! - `infer + get_results`: one synchronous round trip
//! - `infer` and `get_results` measured separately
//! - pipelined `infer_async`: `depth` requests in flight, then the results collected
//!
//! Inputs are filled with a fixed pseudo-random pattern; the outputs are not
//! checked. Dynamic input dimensions are set to 1x3x480x640 positions as needed.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rsinference::engine::{Engine, EngineType};
use rsinference::types::{Dtype, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().unwrap_or_else(|| "models/version-RFB-640.onnx".into()));
    let iterations: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(1000);
    let warmup: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(100);
    let backend = match args.next().as_deref().unwrap_or("tensorrt") {
        "tensorrt" => EngineType::TensorRT,
        "openvino" => EngineType::OpenVINO,
        other => return Err(format!("unknown backend '{other}' (use tensorrt or openvino)").into()),
    };
    let depth: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4).max(1);
    let config = args.next();

    let engine = Engine::load(backend)?;
    let start = Instant::now();
    let runner = engine.create_infer_instance(&model, config.as_deref())?;
    if let Some(config) = &config {
        println!("config:    {config}");
    }
    println!("backend:   {backend}");
    println!("model:     {}", model.display());
    println!("create:    {:.1} ms (includes engine build if no cache)", ms(start.elapsed()));

    let inputs: Vec<Tensor> = runner
        .inputs_info()?
        .into_iter()
        .map(|info| {
            let fallback = [1i64, 3, 480, 640];
            let shape: Vec<i64> = info
                .shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if d < 0 { *fallback.get(i).unwrap_or(&1) } else { d })
                .collect();
            let elems: usize = shape.iter().map(|&d| d as usize).product();
            let bytes = elems * dtype_size(info.dtype);
            let mut state = 0x2545_f491_u32;
            let data = (0..bytes)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    // Every byte < 0x40, so each f32's top byte keeps it small and finite.
                    (state & 0x3f) as u8
                })
                .collect();
            println!("input:     {} {:?} {:?} ({:.2} MB)", info.name, shape, info.dtype, bytes as f64 / 1e6);
            Tensor { name: info.name, shape, dtype: info.dtype, data }
        })
        .collect();
    for out in runner.outputs_info()? {
        println!("output:    {} {:?} {:?}", out.name, out.shape, out.dtype);
    }

    for _ in 0..warmup {
        runner.infer(&inputs)?;
        runner.get_results()?;
    }

    let mut round_trip = Vec::with_capacity(iterations);
    let mut infer_only = Vec::with_capacity(iterations);
    let mut results_only = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        runner.infer(&inputs)?;
        let t1 = Instant::now();
        let out = runner.get_results()?;
        let t2 = Instant::now();
        std::hint::black_box(out);
        round_trip.push(t2 - t0);
        infer_only.push(t1 - t0);
        results_only.push(t2 - t1);
    }

    println!();
    println!("{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "sync (ms)", "min", "p50", "p90", "p99", "max", "mean");
    report("infer + get_results", &mut round_trip);
    report("  infer", &mut infer_only);
    report("  get_results", &mut results_only);

    // Pipelined async: keep `depth` requests in flight per batch.
    let batches = (iterations / depth).max(1);
    let mut per_request = Vec::with_capacity(batches);
    let wall = Instant::now();
    for _ in 0..batches {
        let t0 = Instant::now();
        for _ in 0..depth {
            runner.infer_async(&inputs)?;
        }
        for _ in 0..depth {
            std::hint::black_box(runner.get_results()?);
        }
        per_request.push((Instant::now() - t0) / depth as u32);
    }
    let wall = wall.elapsed();
    let total = batches * depth;
    println!();
    println!("{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", format!("async depth {depth} (ms/req)"), "min", "p50", "p90", "p99", "max", "mean");
    report("infer_async + get_results", &mut per_request);
    println!("throughput:                  {:.1} req/s ({total} requests)", total as f64 / wall.as_secs_f64());
    Ok(())
}

fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 | Dtype::I32 => 4,
        Dtype::I64 => 8,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn report(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let pct = |p: f64| ms(samples[((samples.len() - 1) as f64 * p).round() as usize]);
    let mean = ms(samples.iter().sum::<Duration>()) / samples.len() as f64;
    println!(
        "{:<28} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        label,
        ms(samples[0]),
        pct(0.5),
        pct(0.9),
        pct(0.99),
        ms(samples[samples.len() - 1]),
        mean
    );
}
