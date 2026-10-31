//! CLI entry point: load an engine and report its backend bridge.

use rsinference::engine::{Engine, EngineType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.engine".to_string());

    let engine = Engine::load(path, EngineType::TensorRT)?;
    println!(
        "engine: {} ({})",
        engine.path().display(),
        engine.engine_type()
    );
    println!("bridge: {}", engine.bridge().path.display());

    Ok(())
}
