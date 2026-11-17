//! Links the `ed`, `entanglement`, `lindblad_entanglement` and `state_hist_ed` binaries against Intel MKL (LAPACKE).
//! The library directory is $MKL_LIB_DIR, or ~/anaconda3/lib by default; it is also written as the binaries' rpath.
//! The tests link MKL only if libmkl_rt.so is found there, and then set `cfg(mkl)` (the test of `src/mkl_ed.rs`
//! runs only then). Nothing else links MKL.

fn main() {
    println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");
    let dir = std::env::var("MKL_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/anaconda3/lib", std::env::var("HOME").expect("HOME is not set")));
    for bin in ["ed", "entanglement", "lindblad_entanglement", "state_hist_ed"] {
        println!("cargo:rustc-link-arg-bin={bin}=-L{dir}");
        println!("cargo:rustc-link-arg-bin={bin}=-Wl,-rpath,{dir}");
        println!("cargo:rustc-link-arg-bin={bin}=-lmkl_rt");
    }
    println!("cargo::rustc-check-cfg=cfg(mkl)");
    if std::path::Path::new(&dir).join("libmkl_rt.so").exists() {
        println!("cargo:rustc-cfg=mkl");
        println!("cargo:rustc-link-arg-tests=-L{dir}");
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
        println!("cargo:rustc-link-arg-tests=-lmkl_rt");
    }
}
