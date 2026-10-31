//! Bindgen-generated C types for the bridge (see `build.rs`).
//!
//! Only the type declarations are used. The exported functions are resolved at
//! runtime in `crate::dll`.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    non_upper_case_globals,
    dead_code,
    clippy::all
)]

include!(concat!(env!("OUT_DIR"), "/ffi.rs"));