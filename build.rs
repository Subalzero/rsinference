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
        // Default output directories of the documented Ninja builds.
        PathBuf::from("cpp/build/x64-release"),
        PathBuf::from("cpp/build/linux-release"),
        PathBuf::from("cpp/install/x64-release/bin"),
        PathBuf::from("cpp/build/x64-release/bin"),
        PathBuf::from("cpp/build/bin"),
    ]);

    // With no explicit INFERTENSORRT_BIN_DIR and no local bridge build (the usual case
    // for a crate pulled in as a git dependency), fetch the prebuilt bridge from the
    // GitHub Release that matches this crate's version.
    let find = |dirs: &[PathBuf], file_name: &str| {
        dirs.iter()
            .map(|dir| dir.join(file_name))
            .find(|dll| dll.exists())
    };
    if env::var_os("INFERTENSORRT_BIN_DIR").is_none()
        && find(&candidates, &library_file_name(DLL_NAMES[0])).is_none()
    {
        if let Some(dir) = prebuilt::fetch(&out_path) {
            candidates.push(dir);
        }
    }

    for name in DLL_NAMES {
        let file_name = library_file_name(name);

        // Candidates are in priority order: the first match wins, so
        // INFERTENSORRT_BIN_DIR overrides the default build trees.
        match find(&candidates, &file_name) {
            Some(dll) => {
                // Re-run (and re-copy) when the bridge is rebuilt.
                println!("cargo:rerun-if-changed={}", dll.display());
                copy_next_to_artifacts(&dll, &file_name, &out_path);
            }
            None => {
                println!(
                    "cargo:warning={file_name} not found; the crate builds, but loading this \
                     bridge will fail at runtime (download a release archive and set \
                     INFERTENSORRT_BIN_DIR to its lib directory, or build the C++ bridge)"
                );
            }
        }
    }
}

/// `infertensorrt` → `infertensorrt.dll` / `libinfertensorrt.so` for the host.
fn library_file_name(name: &str) -> String {
    format!(
        "{}{}{}",
        std::env::consts::DLL_PREFIX,
        name,
        std::env::consts::DLL_SUFFIX
    )
}

/// Downloads the prebuilt bridge libraries from the GitHub Release `v<crate version>`.
///
/// Environment variables:
/// - `RSINFERENCE_NO_DOWNLOAD=1` skips the download (offline builds).
/// - `RSINFERENCE_RELEASE_URL` replaces the release download URL, e.g. a mirror
///   holding the same archives and `SHA256SUMS`.
///
/// Every archive is checked against the release's `SHA256SUMS` before it is unpacked.
/// A failed download only warns, so the crate still builds.
mod prebuilt {
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    /// Largest archive or checksum file accepted from the server.
    const MAX_DOWNLOAD: u64 = 512 * 1024 * 1024;

    /// Returns the directory holding the bridge libraries, or `None` (after a warning)
    /// when the download is disabled, unsupported, or failed.
    pub fn fetch(out_dir: &Path) -> Option<PathBuf> {
        println!("cargo:rerun-if-env-changed=RSINFERENCE_NO_DOWNLOAD");
        println!("cargo:rerun-if-env-changed=RSINFERENCE_RELEASE_URL");

        if env::var("RSINFERENCE_NO_DOWNLOAD").is_ok_and(|v| !v.is_empty() && v != "0") {
            return None;
        }
        // docs.rs builds without network access and never runs the code.
        if env::var_os("DOCS_RS").is_some() {
            return None;
        }

        let target = env::var("TARGET").unwrap_or_default();
        let Some(extension) = archive_extension(&target) else {
            println!(
                "cargo:warning=no prebuilt bridge is published for {target}; build the C++ \
                 bridge and set INFERTENSORRT_BIN_DIR"
            );
            return None;
        };

        let version = env!("CARGO_PKG_VERSION");
        let stem = format!("rsinference-v{version}-{target}");
        let lib_dir = out_dir.join("prebuilt").join(&stem).join("lib");
        if lib_dir.is_dir() {
            return Some(lib_dir);
        }

        let base = env::var("RSINFERENCE_RELEASE_URL").unwrap_or_else(|_| {
            format!(
                "{}/releases/download/v{version}",
                env!("CARGO_PKG_REPOSITORY")
            )
        });
        let base = base.trim_end_matches('/');
        let archive_name = format!("{stem}.{extension}");

        match download_and_unpack(base, &archive_name, out_dir) {
            Ok(()) if lib_dir.is_dir() => {
                println!("cargo:warning=using prebuilt bridge from {base}/{archive_name}");
                Some(lib_dir)
            }
            Ok(()) => {
                println!("cargo:warning={archive_name} has no {stem}/lib directory");
                None
            }
            Err(err) => {
                println!(
                    "cargo:warning=could not download the prebuilt bridge ({err}). Set \
                     INFERTENSORRT_BIN_DIR to a local copy, or RSINFERENCE_NO_DOWNLOAD=1 to \
                     stop trying"
                );
                // A path that never exists makes Cargo re-run this script on the next
                // build, so the download is retried once the network is back.
                println!("cargo:rerun-if-changed={}", lib_dir.display());
                None
            }
        }
    }

    /// Archive format the release workflow uses for each supported target.
    fn archive_extension(target: &str) -> Option<&'static str> {
        match target {
            "x86_64-unknown-linux-gnu" => Some("tar.gz"),
            "x86_64-pc-windows-msvc" => Some("zip"),
            _ => None,
        }
    }

    fn download_and_unpack(base: &str, archive_name: &str, out_dir: &Path) -> Result<()> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(300)))
            .build()
            .into();

        let sums = String::from_utf8(get(&agent, &format!("{base}/SHA256SUMS"))?)?;
        let expected = expected_sha256(&sums, archive_name)
            .ok_or_else(|| format!("SHA256SUMS has no entry for {archive_name}"))?;

        let archive = get(&agent, &format!("{base}/{archive_name}"))?;
        let actual = hex(&Sha256::digest(&archive));
        if !actual.eq_ignore_ascii_case(&expected) {
            return Err(
                format!("checksum mismatch for {archive_name}: expected {expected}, got {actual}")
                    .into(),
            );
        }

        // Unpack into a scratch directory first so an interrupted build never leaves a
        // half-extracted lib directory that later builds would trust.
        let prebuilt = out_dir.join("prebuilt");
        let staging = out_dir.join("prebuilt.partial");
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging)?;
        if archive_name.ends_with(".zip") {
            zip::ZipArchive::new(Cursor::new(archive))?.extract(&staging)?;
        } else {
            tar::Archive::new(flate2::read::GzDecoder::new(Cursor::new(archive)))
                .unpack(&staging)?;
        }
        let _ = fs::remove_dir_all(&prebuilt);
        fs::rename(&staging, &prebuilt)?;
        Ok(())
    }

    fn get(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>> {
        agent
            .get(url)
            .call()
            .and_then(|mut response| {
                response
                    .body_mut()
                    .with_config()
                    .limit(MAX_DOWNLOAD)
                    .read_to_vec()
            })
            .map_err(|err| format!("{url}: {err}").into())
    }

    /// Finds `name` in `sha256sum` output (`<hex>  <name>` or `<hex> *<name>`).
    fn expected_sha256(sums: &str, name: &str) -> Option<String> {
        sums.lines().find_map(|line| {
            let (hash, file) = line.trim().split_once(char::is_whitespace)?;
            let file = file.trim_start().trim_start_matches('*');
            (file == name).then(|| hash.to_string())
        })
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
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
