//! CLI entry point: load an engine and report its backend bridge.

use rsinference::engine::{Engine, EngineType};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let engine = Engine::load(EngineType::TensorRT)?;
    println!(
        "engine: {}",
        engine.engine_type()
    );

    Ok(())
}
