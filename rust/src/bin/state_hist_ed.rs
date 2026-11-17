//! γ = 0 diagonal-ensemble Bloch vectors n̄_j of bulk product states, for a(t) in `state_hist`:
//!
//!     cargo run --release --bin state_hist_ed -- --L 12 --init-seeds 1-100 [--seed 1]
//!
//! H (random c, realization --seed) is diagonalized densely with zheevd (MKL, linked by build.rs; 16 · 4^L bytes,
//! L ≤ 13); for every init seed N the bulk product state of `initial::bulk_product` (the same state as
//! `state_hist --init bulk-product --init-seed N`) gives n̄^α_j = Σ_E |⟨E|ψ⟩|² ⟨E|σ^α_j|E⟩.
//!
//! Output (data/ or $SCARS_DATA_DIR), one file per init seed, skipped if it exists:
//!   nbar_L<L>[_seed<N>]_initbulk-product_s<init seed>.csv   first line `# L=… E=… E_ed=… norm=… …`, then j, nx, ny, nz

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use open_exact_scars::cli::Options;
use open_exact_scars::initial::{bulk_product, Init};
use open_exact_scars::model::{seed_suffix, Params};
use open_exact_scars::trajectory::{Hamiltonian, Rng};

#[path = "../mkl_ed.rs"]
mod mkl_ed;

fn main() {
    let args = Options::parse(&["L", "init-seeds", "seed"]);
    let l = args.usize("L", 0);
    assert!((4..=13).contains(&l), "--L between 4 and 13");
    let seed = args.usize("seed", 1) as u64;
    let seeds: Vec<u64> = match args.str("init-seeds").expect("--init-seeds a-b or a,b,c").split_once('-') {
        Some((a, b)) => (a.parse().unwrap()..=b.parse().unwrap()).collect(),
        None => args.str("init-seeds").unwrap().split(',').map(|s| s.trim().parse().unwrap()).collect(),
    };
    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");
    let file = |s: u64| data.join(format!("nbar_L{l}{}_{}.csv", seed_suffix(seed), Init::BulkProduct { seed: s }.tag()));
    let todo: Vec<u64> = seeds.into_iter().filter(|&s| !file(s).exists()).collect();
    if todo.is_empty() {
        println!("L = {l}: every n̄ file exists");
        return;
    }

    let start = Instant::now();
    let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
    let ham = Hamiltonian::new(&params);
    let eig = mkl_ed::Eigen::new(&params);
    println!("L = {l}: H diagonalized in {:.1?}", start.elapsed());
    for s in todo {
        let bulk = bulk_product(&ham, s);
        let (nbar, norm, e_ed) = eig.diagonal_ensemble(&bulk.psi);
        let norm_perp = nbar.iter().map(|n| n[0].hypot(n[1])).sum::<f64>() / l as f64;
        let tmp = data.join(format!("nbar_L{l}_s{s}.part.{}", std::process::id()));
        let mut csv = BufWriter::new(File::create(&tmp).unwrap());
        writeln!(
            csv,
            "# L={l} seed={seed} init_seed={s} E={:.12e} E_ed={e_ed:.12e} norm={norm:.15} nbar_norm={norm_perp:.6e}",
            bulk.energy
        )
        .unwrap();
        writeln!(csv, "j,nx,ny,nz").unwrap();
        for (j, n) in nbar.iter().enumerate() {
            writeln!(csv, "{j},{:.15e},{:.15e},{:.15e}", n[0], n[1], n[2]).unwrap();
        }
        csv.into_inner().unwrap().sync_all().unwrap();
        fs::rename(&tmp, file(s)).unwrap();
        println!("  init seed {s}: E/L = {:.4} (ED {:.4}), Σ|c|² − 1 = {:.1e}, (1/L) Σ |n̄⊥_j| = {norm_perp:.4}", bulk.energy / l as f64, e_ed / l as f64, norm - 1.0);
    }
    println!("L = {l}: done in {:.1?}", start.elapsed());
}
