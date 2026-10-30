//! Build script: generates Rust bindings for the C++ bridge in `cpp/`.
//!
//! Bindings are only generated when the `ffi` feature is enabled and the
//! bridge header exists, so plain `cargo build` works without TensorRT.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/include");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("CARGO_FEATURE_FFI").is_none() {
        return;
    }

    let header_path = PathBuf::from("cpp/include").join("infer_bridge.h");
    if !header_path.exists() {
        println!(
            "cargo:warning=cpp/include/infer_bridge.h not found; skipping bindgen \
             (build the C++ bridge first, e.g. cmake -S cpp -B cpp/build)"
        );
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ffi.rs"))
        .expect("couldn't write bindings");

    // Link against the built bridge library.
    println!("cargo:rustc-link-search=native=cpp/build");
    println!("cargo:rustc-link-lib=static=infer_bridge");
}
