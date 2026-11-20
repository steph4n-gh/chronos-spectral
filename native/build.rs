//! Records the build configuration in the `chronos_replay` binary, so `--bench` output states
//! exactly what was timed: the Cargo profile directory (`release`, `checked`, `profiling`, ...),
//! the opt-level, the debug-info setting and the `-C target-cpu` passed in rustflags, if any.
//! Nothing here affects code generation.

fn main() {
    let out = std::env::var("OUT_DIR").unwrap_or_default();
    // OUT_DIR = <target-dir>/[<triple>/]<profile>/build/<pkg>-<hash>/out
    let parts: Vec<&str> = out.split(['/', '\\']).collect();
    let profile = parts
        .iter()
        .rposition(|p| *p == "build")
        .and_then(|i| i.checked_sub(1))
        .map(|i| parts[i])
        .unwrap_or("?");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let flags: Vec<&str> = flags.split('\x1f').collect();
    let mut cpu = String::from("default");
    for (i, f) in flags.iter().enumerate() {
        if let Some(v) = f.strip_prefix("target-cpu=") {
            cpu = v.to_owned();
        } else if let Some(v) = f.strip_prefix("-Ctarget-cpu=") {
            cpu = v.to_owned();
        } else if *f == "-C" {
            if let Some(v) = flags.get(i + 1).and_then(|n| n.strip_prefix("target-cpu=")) {
                cpu = v.to_owned();
            }
        }
    }
    println!("cargo:rustc-env=CHRONOS_PROFILE={profile}");
    println!(
        "cargo:rustc-env=CHRONOS_OPT_LEVEL={}",
        std::env::var("OPT_LEVEL").unwrap_or_default()
    );
    println!("cargo:rustc-env=CHRONOS_TARGET_CPU={cpu}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
