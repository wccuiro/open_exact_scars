//! Floquet circuit (src/floquet.rs) for ρ from |−⟩^⊗L with depolarizing noise D(ρ) = γ Σ_j Σ_α (σ^α_j ρ σ^α_j − ρ)
//! (jumps √γ σ^α, `--channel xyz`: every Pauli decays at 4γ; since 2026-11-26, before `dep` √(γ/2) σ^α, whose γ is
//! twice this one: output without the channel column), no kick: singlet weight ⟨R_{j,j+1}⟩ = (1 − ⟨XX⟩ − ⟨YY⟩ −
//! ⟨ZZ⟩)/4 on every bond of the ring after every cycle.
//! From rust/: cargo run --release --example floquet_bond -- [L] [gamma] [dt] [cycles] > ../data/floquet_bond.csv
use std::io::{self, BufWriter, Write};

use open_exact_scars::floquet::{product_state, Floquet};
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::Rng;

fn main() -> io::Result<()> {
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().expect("numbers")).collect();
    let get = |i: usize, d: f64| args.get(i).copied().unwrap_or(d);
    let (l, gamma, dt, cycles) = (get(0, 12.0) as usize, get(1, 0.001), get(2, 0.1), get(3, 1000.0) as usize);
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    let channel = Channel::from_options(Some("xyz"), None, 1.0);
    let floquet = Floquet::new(&params, &channel, gamma, dt);
    let mut r = product_state(&vec![[-1.0, 0.0, 0.0]; l]);
    // r_P = Tr(ρ P), index x | z << L; σˣσˣ: x = both bits; σᶻσᶻ: z = both bits; σʸσʸ: both.
    let corr = |r: &[f64], j: usize| {
        let m = (1usize << j) | (1usize << ((j + 1) % l));
        [r[m], r[m | m << l], r[m << l]]
    };
    eprintln!("L = {l}, gamma = {gamma}, dt = {dt}, {cycles} cycles");
    let mut csv = BufWriter::new(io::stdout().lock());
    writeln!(csv, "L,gamma,dt,cycle,t,bond,singlet,xx,yy,zz,channel")?;
    for n in 0..=cycles {
        if n > 0 {
            floquet.step(&mut r);
        }
        for j in 0..l {
            let [xx, yy, zz] = corr(&r, j);
            let singlet = (1.0 - xx - yy - zz) / 4.0;
            writeln!(csv, "{l},{gamma},{dt},{n},{:.4},{j},{singlet:.12e},{xx:.12e},{yy:.12e},{zz:.12e},xyz", n as f64 * dt)?;
        }
        if n % 100 == 0 {
            eprintln!("cycle {n}");
        }
    }
    csv.flush()
}
