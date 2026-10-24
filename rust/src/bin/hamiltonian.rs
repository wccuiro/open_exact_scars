//! Exports H as a list of Pauli strings, for the Julia codes (CLIK, `julia/clik.jl`).
//!
//!     cargo run --release --bin hamiltonian -- [--L 6 | --lmin 4 --lmax 8] [--seed 1] [--open 0|1]
//!
//! H: src/parameters.rs with random c (Params::random_c, generator seeded with 1000 · seed + L, the same
//! realization as `states`, `ed` and `lindblad_entanglement`). --open 1: open chain (terms that leave 0 … L−1
//! dropped) instead of the periodic ring of parameters.rs; same c realization.
//!
//! Output (data/ or $SCARS_DATA_DIR): hamiltonian_L<L>.csv (hamiltonian_open_L<L>.csv with --open 1), columns
//!   string, coef
//! one row per Pauli string, H = Σ coef · string; `string` has one character I, X, Y, Z per site 0 … L−1.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use open_exact_scars::cli::Options;
use open_exact_scars::model::{hamiltonian_terms, Params};
use open_exact_scars::trajectory::Rng;

fn main() {
    let args = Options::parse(&["L", "lmin", "lmax", "seed", "open"]);
    let open_chain = args.usize("open", 0) == 1;
    let sizes = args.sizes(4, 8);
    let seed = args.usize("seed", 1) as u64;

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        let mut params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        if open_chain {
            params.periodic = false;
        }
        let terms = hamiltonian_terms(&params);
        let file = data.join(format!("hamiltonian{}_L{l}.csv", if open_chain { "_open" } else { "" }));
        let mut csv = BufWriter::new(File::create(&file).unwrap());
        writeln!(csv, "string,coef").unwrap();
        for (p, coef) in &terms {
            let string: String = (0..l)
                .map(|j| match ((p.x >> j) & 1, (p.z >> j) & 1) {
                    (0, 0) => 'I',
                    (1, 0) => 'X',
                    (1, 1) => 'Y',
                    _ => 'Z',
                })
                .collect();
            writeln!(csv, "{string},{coef:.17e}").unwrap();
        }
        println!("L = {l}: {} Pauli strings -> {}", terms.len(), file.display());
    }
}
