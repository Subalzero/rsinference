//! Build script: generates Rust bindings for the C++ bridge in `cpp/`.
//!
//! The bridge is a DLL that is loaded **at runtime** (see `src/dll.rs`), so this
//! script never emits `cargo:rustc-link-lib`: bindgen only produces the type
//! declarations, and the exported functions are resolved with `libloading`.
//!
//! The native bridge is not optional: `src/ffi.rs` unconditionally includes the
//! generated bindings, so a missing header is a hard build error.

use std::env;
use std::path::{Path, PathBuf};

/// CMake targets that produce bridge DLLs, as named by
/// `EngineType::bridge_name` in `src/engine.rs`. `inferopenvino` is only
/// built when CMake finds OpenVINO.
const DLL_NAMES: &[&str] = &["infertensorrt", "inferopenvino"];

fn main() {
    println!("cargo:rerun-if-changed=cpp/include");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=INFERTENSORRT_BIN_DIR");

    // The public C-compatible surface lives in cpp/include/public/rsinference.h.
    let header_path = PathBuf::from("cpp/include")
        .join("public")
        .join("rsinference.h");
    if !header_path.exists() {
        panic!(
            "{} not found: the native bridge is required. Build the C++ bridge first, e.g. \
             `cmake -S cpp -B cpp/build/x64-release`.",
            header_path.display()
        );
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

    // Locate the built bridge: the runtime loader needs the shared library
    // itself, not an import library. Set INFERTENSORRT_BIN_DIR to point at
    // another build tree.
    let mut candidates = Vec::new();
    if let Ok(dir) = env::var("INFERTENSORRT_BIN_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    candidates.extend([
        // Default output directory used by the documented Ninja build.
        PathBuf::from("cpp/build/x64-release"),
        PathBuf::from("cpp/install/x64-release/bin"),
        PathBuf::from("cpp/build/x64-release/bin"),
        PathBuf::from("cpp/build/bin"),
    ]);

    for name in DLL_NAMES {
        let file_name = format!(
            "{}{}{}",
            std::env::consts::DLL_PREFIX,
            name,
            std::env::consts::DLL_SUFFIX
        );

        // Candidates are in priority order: the first match wins, so
        // INFERTENSORRT_BIN_DIR overrides the default build trees.
        let dll = candidates
            .iter()
            .map(|dir| dir.join(&file_name))
            .find(|dll| dll.exists());

        match dll {
            Some(dll) => {
                // Re-run (and re-copy) when the bridge is rebuilt.
                println!("cargo:rerun-if-changed={}", dll.display());
                copy_next_to_artifacts(&dll, &file_name, &out_path);
            }
            None => {
                println!(
                    "cargo:warning={file_name} not found; the crate builds, but loading this \
                     bridge will fail at runtime (install the C++ bridge, e.g. \
                     `cmake --install cpp/build/x64-release --prefix cpp/install`)"
                );
            }
        }
    }
}

/// Copy the bridge next to the built artifacts so `Bridge::load_backend`
/// resolves it through the exe directory.
fn copy_next_to_artifacts(dll: &Path, file_name: &str, out_path: &Path) {
    // `target/<profile>` holds the binaries; `target/<profile>/deps` holds
    // test and bench executables. OUT_DIR is
    // target/<profile>/build/<pkg>-<hash>/out, so the profile directory is
    // three levels up.
    let Some(profile_dir) = out_path.ancestors().nth(3) else {
        return;
    };
    for (dir, announce) in [
        (profile_dir.to_path_buf(), true),
        (profile_dir.join("deps"), false),
    ] {
        if !dir.exists() && std::fs::create_dir_all(&dir).is_err() {
            continue;
        }
        let target = dir.join(file_name);
        match std::fs::copy(dll, &target) {
            Ok(_) if announce => println!(
                "cargo:warning=copied {} to {}",
                file_name,
                target.to_string_lossy()
            ),
            Ok(_) => {}
            Err(err) => println!(
                "cargo:warning=could not copy {} to {} ({err}); put the bridge's \
                 directory on PATH",
                file_name,
                target.to_string_lossy()
            ),
        }
    }
}
