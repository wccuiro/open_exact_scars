//! Links the `ed`, `entanglement` and `lindblad_entanglement` binaries against Intel MKL (LAPACKE). The library directory is
//! $MKL_LIB_DIR, or ~/anaconda3/lib by default; it is also written as the binaries' rpath. Nothing else links MKL.

fn main() {
    println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");
    let dir = std::env::var("MKL_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/anaconda3/lib", std::env::var("HOME").expect("HOME is not set")));
    for bin in ["ed", "entanglement", "lindblad_entanglement"] {
        println!("cargo:rustc-link-arg-bin={bin}=-L{dir}");
        println!("cargo:rustc-link-arg-bin={bin}=-Wl,-rpath,{dir}");
        println!("cargo:rustc-link-arg-bin={bin}=-lmkl_rt");
    }
}
