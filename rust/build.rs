//! Links the `ed` binary against Intel MKL (LAPACKE). The library directory is $MKL_LIB_DIR, or
//! ~/anaconda3/lib by default; it is also written as the binary's rpath. Nothing else links MKL.

fn main() {
    println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");
    let dir = std::env::var("MKL_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/anaconda3/lib", std::env::var("HOME").expect("HOME is not set")));
    println!("cargo:rustc-link-arg-bin=ed=-L{dir}");
    println!("cargo:rustc-link-arg-bin=ed=-Wl,-rpath,{dir}");
    println!("cargo:rustc-link-arg-bin=ed=-lmkl_rt");
}
