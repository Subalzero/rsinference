//! CLI entry point: load an engine and run one inference.

use infertensorrt_rs::engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.engine".to_string());

    let engine = Engine::load(path)?;
    println!("engine: {}", engine.path.display());

    let input = vec![0.0f32; engine.input.dims.iter().product::<usize>()];
    let output = engine.infer(&input)?;
    println!("output: {output:?}");

    Ok(())
}
