//! Build script: generates Rust bindings for the C++ bridge in `cpp/`.
//!
//! The bridge is a DLL that is loaded **at runtime** (see `src/dll.rs`), so this
//! script never emits `cargo:rustc-link-lib`: bindgen only produces the type
//! declarations, and the exported functions are resolved with `libloading`.
//!
//! Bindings are only generated when the `ffi` feature is enabled and the bridge
//! header exists, so plain `cargo build` works without TensorRT.

use std::env;
use std::path::PathBuf;

/// Name of the CMake target that produces the bridge DLL.
const DLL_NAME: &str = "infertensorrt_cpp";

fn main() {
    println!("cargo:rerun-if-changed=cpp/include");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=INFERTENSORRT_BIN_DIR");

    if env::var_os("CARGO_FEATURE_FFI").is_none() {
        return;
    }

    // The public C-compatible surface lives in cpp/include/public/rsinference.h.
    let header_path = PathBuf::from("cpp/include")
        .join("public")
        .join("rsinference.h");
    if !header_path.exists() {
        println!(
            "cargo:warning=cpp/include/public/rsinference.h not found; skipping bindgen \
             (build the C++ bridge first, e.g. cmake -S cpp -B cpp/build/x64-release)"
        );
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        // The header is C++ (default member initializers, __declspec(dllexport)),
        // so parse it as C++ and let the extern "C" block define the linkage.
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++23")
        // Default enum style (plain consts + integer alias) keeps RSInferDataType
        // ABI-compatible with the C `enum`, which MSVC lays out as a 4-byte int.
        // The generated `extern "C" fn` declarations are unused: `src/dll.rs`
        // resolves the symbols dynamically, so nothing is linked against.
        .allowlist_type("RSInfer.*")
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ffi.rs"))
        .expect("couldn't write bindings");

    // Locate the built DLL: the runtime loader needs the binary itself, not an
    // import library. Set INFERTENSORRT_BIN_DIR to point at another build tree.
    let candidates = [
        env::var_os("INFERTENSORRT_BIN_DIR").map(PathBuf::from),
        Some(PathBuf::from("cpp/install/x64-release/bin")),
        Some(PathBuf::from("cpp/build/x64-release/bin")),
        Some(PathBuf::from("cpp/build/bin")),
    ];

    let dll = candidates
        .into_iter()
        .flatten()
        .map(|dir| dir.join(format!("{DLL_NAME}.dll")))
        .find(|dll| dll.exists());

    match dll {
        Some(dll) => {
            // Baked into the crate at compile time; `src/dll.rs` reads it with
            // option_env! so a missing path degrades to a runtime error instead
            // of a compile error.
            println!(
                "cargo:rustc-env=INFERTENSORRT_DLL={}",
                dll.to_string_lossy()
            );

            // Copy the DLL next to the executable so a bare `Library::new(
            // "infertensorrt_cpp.dll")` resolves via the exe's own directory.
            // OUT_DIR is target/<profile>/build/<pkg>-<hash>/out, so the profile
            // directory is three levels up.
            if let Some(profile_dir) = out_path.ancestors().nth(3) {
                let target = profile_dir.join(format!("{DLL_NAME}.dll"));
                match std::fs::copy(&dll, &target) {
                    Ok(_) => println!(
                        "cargo:warning=copied {} to {}",
                        DLL_NAME,
                        target.to_string_lossy()
                    ),
                    Err(err) => println!(
                        "cargo:warning=could not copy {} to {} ({err}); put the DLL's \
                         directory on PATH",
                        DLL_NAME,
                        target.to_string_lossy()
                    ),
                }
            }
        }
        None => {
            println!(
                "cargo:warning={DLL_NAME}.dll not found; the ffi bindings are generated but \
                 the runtime loader will fail (install the C++ bridge, e.g. cmake --install \
                 cpp/build/x64-release --prefix cpp/install)"
            );
        }
    }
}
