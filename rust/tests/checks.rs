//! The two exact references of CLAUDE.md. Run with `cargo test --release`.

use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator, Sector};
use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{
    coherences, random_direction, reference_bloch, Hamiltonian, JumpTrajectory, Rng, Trajectory,
};

/// Largest deviation of (m_x, m_y, m_z) from (−cos ht, −sin ht, 0) · e^(−2γt) up to t = 30.
fn deviation(p: &Params, gamma: f64) -> f64 {
    let liou = Liouvillian::new(p);
    let magnetization = Magnetization::new(&liou);
    let mut v = liou.initial_state();
    let dt = 0.1;
    let mut prop = Propagator::new(&liou, gamma, dt);
    let mut worst = 0.0f64;
    for step in 1..=300 {
        prop.step(&mut v);
        let t = step as f64 * dt;
        let decay = (-2.0 * gamma * t).exp();
        let [mx, my, mz] = magnetization.measure(&v);
        worst = worst.max((mx + (p.h * t).cos() * decay).abs()).max((my + (p.h * t).sin() * decay).abs()).max(mz.abs());
    }
    worst
}

/// γ = 0, full H: the state stays in the tower and the oscillations are perfect.
#[test]
fn reference_1_noiseless_oscillations_are_perfect() {
    assert!(deviation(&Params::baseline(6), 0.0) < 1e-10);
}

/// H_I = 0 with noise: independent spins, the oscillations decay as e^(−2γt).
#[test]
fn reference_2_free_spins_with_dephasing() {
    assert!(deviation(&Params::free_spins(4), 0.1) < 1e-11);
    assert!(deviation(&Params::free_spins(6), 0.1) < 1e-11);
}

/// H_I = 0 with noise, from a product state with random Bloch vectors: every spin precesses about z
/// and loses its transverse part as e^(−2γt), so m_x + i m_y = (m_x(0) + i m_y(0)) e^{iht} e^(−2γt)
/// and m_z stays constant.
#[test]
fn free_spins_from_a_random_product_state() {
    let p = Params::free_spins(5);
    let mut rng = Rng::new(7);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let liou = Liouvillian::new(&p);
    let magnetization = Magnetization::new(&liou);
    let mut v = liou.product_state(&n);
    let m0 = magnetization.measure(&v);
    let (gamma, dt) = (0.1, 0.1);
    let mut prop = Propagator::new(&liou, gamma, dt);
    let mut worst = 0.0f64;
    for step in 1..=300 {
        prop.step(&mut v);
        let t = step as f64 * dt;
        let (sin, cos) = (p.h * t).sin_cos();
        let decay = (-2.0 * gamma * t).exp();
        let [mx, my, mz] = magnetization.measure(&v);
        worst = worst
            .max((mx - (m0[0] * cos - m0[1] * sin) * decay).abs())
            .max((my - (m0[0] * sin + m0[1] * cos) * decay).abs())
            .max((mz - m0[2]).abs());
    }
    let mean = |a: usize| n.iter().map(|v| v[a]).sum::<f64>() / p.l as f64;
    assert!((0..3).all(|a| (m0[a] - mean(a)).abs() < 1e-14));
    assert!(worst < 1e-11, "deviation {worst:e}");
}

/// The full Pauli basis and the translation sector give the same site-averaged magnetization for the
/// translation-invariant baseline, from a product state that is not translation invariant.
#[test]
fn full_basis_matches_translation_sector() {
    let p = Params::baseline(5);
    let mut rng = Rng::new(3);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let (gamma, dt) = (0.05, 0.1);
    let orbits = Liouvillian::new(&p);
    let full = Liouvillian::with_sector(&p, Sector::full(p.l));
    let (m_orbits, m_full) = (Magnetization::new(&orbits), Magnetization::new(&full));
    let (mut v, mut w) = (orbits.product_state(&n), full.product_state(&n));
    let (mut prop_v, mut prop_w) = (Propagator::new(&orbits, gamma, dt), Propagator::new(&full, gamma, dt));
    let mut worst = 0.0f64;
    for _ in 0..200 {
        prop_v.step(&mut v);
        prop_w.step(&mut w);
        let (a, b) = (m_orbits.measure(&v), m_full.measure(&w));
        worst = (0..3).map(|i| (a[i] - b[i]).abs()).fold(worst, f64::max);
    }
    assert!(worst < 1e-11, "deviation {worst:e}");
}

/// Random c (no translation), γ = 0: the scar state still oscillates perfectly on every site.
#[test]
fn random_c_keeps_the_scar_oscillations() {
    let p = Params::random_c(6, &mut Rng::new(11));
    let liou = Liouvillian::new(&p);
    assert_eq!(liou.dim(), 1 << 12, "random c must use the full basis");
    let magnetization = Magnetization::new(&liou);
    let mut v = liou.initial_state();
    let mut prop = Propagator::new(&liou, 0.0, 0.1);
    let mut worst = 0.0f64;
    for step in 1..=300 {
        prop.step(&mut v);
        let (sin, cos) = (p.h * step as f64 * 0.1).sin_cos();
        let [x, y, z] = magnetization.per_site(&v);
        for j in 0..p.l {
            worst = worst.max((x[j] + cos).abs()).max((y[j] + sin).abs()).max(z[j].abs());
        }
    }
    assert!(worst < 1e-10, "deviation {worst:e}");
}

/// The matrix-free Hamiltonian part (full basis, used from L = 13) gives the same generator, norm bound
/// and evolution as the sparse matrix.
#[test]
fn matrix_free_matches_sparse() {
    let p = Params::random_c(6, &mut Rng::new(5));
    let sparse = Liouvillian::with_sector(&p, Sector::full(p.l));
    let free = Liouvillian::matrix_free(&p);
    assert!(free.is_matrix_free() && !sparse.is_matrix_free());
    let gamma = 0.3;
    let (a, b) = (sparse.dense(gamma), free.dense(gamma));
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-14, "generator deviation {worst:e}");
    assert!((sparse.norm_bound(gamma) - free.norm_bound(gamma)).abs() < 1e-12);

    let mut rng = Rng::new(9);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let (mut v, mut w) = (sparse.product_state(&n), free.product_state(&n));
    let (mut prop_v, mut prop_w) = (Propagator::new(&sparse, gamma, 0.1), Propagator::new(&free, gamma, 0.1));
    let m = Magnetization::new(&sparse);
    let mut worst = 0.0f64;
    for _ in 0..100 {
        prop_v.step(&mut v);
        prop_w.step(&mut w);
        let (x, y) = (m.per_site(&v), m.per_site(&w));
        for a in 0..3 {
            worst = x[a].iter().zip(&y[a]).map(|(s, t)| (s - t).abs()).fold(worst, f64::max);
        }
    }
    assert!(worst < 1e-12, "deviation {worst:e}");
}

/// Damage D(τ) = Σ_j [1 − Re c_j] of one σᶻ kick on the scar state at phase φ (as in `single_flip`).
fn damage(p: &Params, phi: f64, kick: Option<usize>, taus: &[f64]) -> Vec<(f64, Vec<[f64; 2]>)> {
    let ham = Hamiltonian::new(p);
    let t0 = phi / p.h;
    let mut traj = Trajectory::product(&ham, &reference_bloch(p.l, p.h, t0, kick), 0);
    let mut tau = 0.0;
    taus.iter()
        .map(|&t| {
            if t > tau {
                traj.unitary(t - tau);
                tau = t;
            }
            let c = coherences(&traj.per_site(), p.h, t0 + tau);
            (c.iter().map(|c| 1.0 - c[0]).sum(), c)
        })
        .collect()
}

/// single_flip: D(0) = 2; with H_I = 0 D stays 2; without the kick every c_j stays 1 (full H, random c).
#[test]
fn single_flip_damage() {
    let taus = [0.0, 0.7, 3.1, 10.0];
    let full = Params::random_c(6, &mut Rng::new(13));
    for (d, _) in damage(&full, 0.0, Some(2), &taus[..1]).into_iter().chain(damage(&full, 1.3, Some(4), &taus[..1])) {
        assert!((d - 2.0).abs() < 1e-12, "D(0) = {d}");
    }
    for (d, _) in damage(&Params::free_spins(6), 2.0, Some(1), &taus) {
        assert!((d - 2.0).abs() < 1e-11, "free spins: D = {d}");
    }
    for (_, c) in damage(&full, 0.9, None, &taus) {
        let worst = c.iter().map(|c| (c[0] - 1.0).abs().max(c[1].abs())).fold(0.0, f64::max);
        assert!(worst < 1e-11, "no kick: |c_j − 1| = {worst:e}");
    }
    // with the kick and H_I ≠ 0 the damage does change
    let d = damage(&full, 0.0, Some(2), &taus);
    assert!((d[3].0 - 2.0).abs() > 1e-3);
}

/// Jump unravelling from the scar state: writing the reference-1 state down at the first kick (analytic
/// start) gives the same trajectory as evolving |−⟩^⊗L with H from t = 0.
#[test]
fn jump_analytic_start_matches_evolution() {
    let p = Params::random_c(6, &mut Rng::new(17));
    let ham = Hamiltonian::new(&p);
    let bloch = reference_bloch(p.l, p.h, 0.0, None);
    let mut worst = 0.0f64;
    for seed in 0..5 {
        let mut a = JumpTrajectory::new(&ham, p.h, 0.3, &bloch, true, seed);
        let mut b = JumpTrajectory::new(&ham, p.h, 0.3, &bloch, false, seed);
        for step in 1..=100 {
            let t = 0.1 * step as f64;
            a.advance_to(t);
            b.advance_to(t);
            let (x, y) = (a.per_site(), b.per_site());
            for k in 0..3 {
                worst = x[k].iter().zip(&y[k]).map(|(s, t)| (s - t).abs()).fold(worst, f64::max);
            }
        }
        assert!(a.kicks > 0 && a.kicks == b.kicks);
    }
    assert!(worst < 1e-11, "deviation {worst:e}");
}

/// Jump unravelling with H_I = 0: the trajectory average of m_x + i m_y is −e^{iht} e^(−2γt) (reference 2),
/// within the statistical error.
#[test]
fn jump_free_spins_decay() {
    let p = Params::free_spins(4);
    let ham = Hamiltonian::new(&p);
    let (gamma, n) = (0.1, 40000);
    let bloch = reference_bloch(p.l, p.h, 0.0, None);
    let mut trajs: Vec<JumpTrajectory> =
        (0..n).map(|k| JumpTrajectory::new(&ham, p.h, gamma, &bloch, true, 1000 + k as u64)).collect();
    let mut worst = 0.0f64;
    for step in 1..=20 {
        let t = 0.5 * step as f64;
        let (mut mx, mut my) = (0.0, 0.0);
        for traj in &mut trajs {
            traj.advance_to(t);
            let site = traj.per_site();
            mx += site[0].iter().sum::<f64>() / p.l as f64;
            my += site[1].iter().sum::<f64>() / p.l as f64;
        }
        let decay = (-2.0 * gamma * t).exp();
        let (sin, cos) = (p.h * t).sin_cos();
        worst = worst.max((mx / n as f64 + cos * decay).abs()).max((my / n as f64 + sin * decay).abs());
    }
    // one trajectory's m_x has spread ≤ 1, so the error of the mean is ≤ 1/√n = 0.005
    assert!(worst < 0.02, "deviation {worst:e}");
}

// ---------------------------------------------------------------------------------------------------------------
// Linear algebra, entanglement, operator statistics, noise channels (lean-data-plan.md, step 2)

use open_exact_scars::entanglement::{
    block_spectrum, block_spectrum_par, entropies, hermitian_eigenvalues, operator_stats, operator_weights,
    sym_eigenvalues,
};
use open_exact_scars::noise::Channel;

fn moments(eigs: &[f64]) -> [f64; 3] {
    [1, 2, 3].map(|k| eigs.iter().map(|e| e.powi(k)).sum())
}

/// Tr A, Tr A², Tr A³ of a dense real n × n matrix.
fn trace_moments(a: &[f64], n: usize) -> [f64; 3] {
    let mul = |x: &[f64], y: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let xik = x[i * n + k];
                for j in 0..n {
                    z[i * n + j] += xik * y[k * n + j];
                }
            }
        }
        z
    };
    let a2 = mul(a, a);
    let a3 = mul(&a2, a);
    let tr = |m: &[f64]| (0..n).map(|i| m[i * n + i]).sum::<f64>();
    [tr(a), tr(&a2), tr(&a3)]
}

#[test]
fn symmetric_eigenvalues_match_traces() {
    let mut rng = Rng::new(21);
    for n in [1, 2, 7, 60, 600] {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let x = rng.gaussian();
                a[i * n + j] = x;
                a[j * n + i] = x;
            }
        }
        let e = sym_eigenvalues(a.clone(), n);
        let (m, t) = (moments(&e), trace_moments(&a, n));
        for k in 0..3 {
            assert!((m[k] - t[k]).abs() < 1e-9 * (1.0 + t[k].abs()), "n = {n}, moment {}: {} vs {}", k + 1, m[k], t[k]);
        }
    }
}

#[test]
fn hermitian_eigenvalues_match_traces() {
    let mut rng = Rng::new(22);
    let n = 40;
    let (mut re, mut im) = (vec![0.0; n * n], vec![0.0; n * n]);
    for i in 0..n {
        for j in 0..=i {
            let (x, y) = (rng.gaussian(), if i == j { 0.0 } else { rng.gaussian() });
            re[i * n + j] = x;
            re[j * n + i] = x;
            im[i * n + j] = y;
            im[j * n + i] = -y;
        }
    }
    let e = hermitian_eigenvalues(&re, &im, n);
    assert_eq!(e.len(), n);
    // moments of M = X + iY through the real embedding [[X, −Y], [Y, X]] (each eigenvalue twice)
    let n2 = 2 * n;
    let mut a = vec![0.0; n2 * n2];
    for i in 0..n {
        for j in 0..n {
            a[i * n2 + j] = re[i * n + j];
            a[(i + n) * n2 + j + n] = re[i * n + j];
            a[i * n2 + j + n] = -im[i * n + j];
            a[(i + n) * n2 + j] = im[i * n + j];
        }
    }
    let (m, t) = (moments(&e), trace_moments(&a, n2));
    for k in 0..3 {
        assert!((2.0 * m[k] - t[k]).abs() < 1e-9 * (1.0 + t[k].abs()));
    }
}

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// Dicke states: block entropies equal the exact formula, for blocks smaller and larger than half the chain.
#[test]
fn dicke_block_entropy() {
    let l = 8;
    for n in [0, 1, 3, 4] {
        let amp = binomial(l, n).powf(-0.5);
        let psi: Vec<[f64; 2]> =
            (0..1usize << l).map(|s| if s.count_ones() as usize == n { [amp, 0.0] } else { [0.0, 0.0] }).collect();
        for ell in 1..l {
            let exact: Vec<f64> = (0..=n).map(|k| binomial(ell, k) * binomial(l - ell, n - k) / binomial(l, n)).collect();
            let (s_exact, s2_exact) = entropies(&exact);
            let (s, s2) = entropies(&block_spectrum(&psi, l, ell));
            assert!((s - s_exact).abs() < 1e-12 && (s2 - s2_exact).abs() < 1e-12, "n = {n}, ℓ = {ell}");
            let (sp, s2p) = entropies(&block_spectrum_par(&psi, l, ell));
            assert!((sp - s).abs() < 1e-12 && (s2p - s2).abs() < 1e-12, "parallel path, n = {n}, ℓ = {ell}");
        }
    }
}

/// Product states and other low-rank ρ_A (many zero eigenvalues): entropy 0 for a product state, and the GHZ state
/// gives ln 2 for every block.
#[test]
fn low_rank_block_spectra() {
    let l = 10;
    let mut rng = Rng::new(35);
    let n: Vec<[f64; 3]> = (0..l).map(|_| random_direction(&mut rng)).collect();
    let p = Params::free_spins(l);
    let ham = Hamiltonian::new(&p);
    let traj = Trajectory::product(&ham, &n, 0);
    // the scar state as the trajectories build it (amplitudes with round-off imaginary parts ≈ 4e-17): its ℓ = 5
    // block once drove the Householder stage into underflow and NaN
    let scar = Trajectory::product(&ham, &vec![[-1.0, 0.0, 0.0]; l], 0);
    let mut ghz = vec![[0.0; 2]; 1 << l];
    ghz[0] = [0.5f64.sqrt(), 0.0];
    ghz[(1 << l) - 1] = [0.0, 0.5f64.sqrt()];
    for ell in 1..l {
        for psi in [&traj.psi, &scar.psi] {
            let (s, s2) = entropies(&block_spectrum(psi, l, ell));
            assert!(s.abs() < 1e-10 && s2.abs() < 1e-10, "product state, ℓ = {ell}: {s}");
        }
        let (s, s2) = entropies(&block_spectrum(&ghz, l, ell));
        let ln2 = 2f64.ln();
        assert!((s - ln2).abs() < 1e-10 && (s2 - ln2).abs() < 1e-10, "GHZ, ℓ = {ell}: {s}");
    }
}

/// A pure entangled ρ (random product state evolved with the full H, γ = 0): the operator entanglement of the exact
/// Pauli-basis ρ is twice the state entanglement of the same state evolved as a vector, for both entropies.
#[test]
fn operator_entanglement_of_a_pure_state() {
    let p = Params::random_c(6, &mut Rng::new(31));
    let mut rng = Rng::new(32);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let liou = Liouvillian::new(&p);
    assert!(liou.is_full());
    let mut v = liou.product_state(&n);
    let mut prop = Propagator::new(&liou, 0.0, 0.5);
    let ham = Hamiltonian::new(&p);
    let mut traj = Trajectory::product(&ham, &n, 0);
    for _ in 0..4 {
        prop.step(&mut v);
        traj.unitary(0.5);
    }
    for ell in [1, 2, 3] {
        let (s_op, s2_op) = entropies(&operator_weights(&v, p.l, ell));
        let (s, s2) = entropies(&block_spectrum(&traj.psi, p.l, ell));
        assert!(s > 0.1, "the state should be entangled (S = {s})");
        assert!((s_op - 2.0 * s).abs() < 1e-8 && (s2_op - 2.0 * s2).abs() < 1e-8, "ℓ = {ell}: {s_op} vs 2 × {s}");
    }
    let st = operator_stats(&v, p.l);
    assert!((st.purity - 1.0).abs() < 1e-10, "pure state: Tr ρ² = {}", st.purity);
    assert!((st.size.iter().sum::<f64>() - 1.0).abs() < 1e-12 && (st.support.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

/// σᶻ dephasing: d ln Tr ρ² / dt = −4γ ⟨n_xy⟩ with weights ∝ r_P² (H conserves Σ r_P²).
#[test]
fn purity_decay_identity() {
    let p = Params::random_c(6, &mut Rng::new(41));
    let liou = Liouvillian::new(&p);
    let gamma = 0.1;
    let mut v = liou.initial_state();
    let mut prop = Propagator::new(&liou, gamma, 0.5);
    let (h, mut fine) = (1e-4, Propagator::new(&liou, gamma, 1e-4));
    for _ in 0..3 {
        prop.step(&mut v);
        let st = operator_stats(&v, p.l);
        let mut w = v.clone();
        fine.step(&mut w);
        let rate = (operator_stats(&w, p.l).purity.ln() - st.purity.ln()) / h;
        assert!((rate + 4.0 * gamma * st.mean_nxy).abs() < 1e-3 * 4.0 * gamma * st.mean_nxy, "{rate} vs {}", -4.0 * gamma * st.mean_nxy);
    }
}

/// Free spins (H_I = 0) with each channel: every site follows its own Bloch equation,
/// dr_x = −h r_y + γ d_x r_x, dr_y = h r_x + γ d_y r_y, dr_z = γ d_z r_z + γ f (d the diagonal, f the feed),
/// integrated here with RK4 and compared with the exact Lindbladian in the full basis.
#[test]
fn channels_with_free_spins() {
    let p = Params::free_spins(4);
    let mut rng = Rng::new(51);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let gamma = 0.2;
    for channel in [Channel::X, Channel::Y, Channel::Depolarizing, Channel::Decay, Channel::Thermal { nbar: 0.5 }] {
        let mut liou = Liouvillian::with_sector(&p, Sector::full(p.l));
        liou.set_channel(channel);
        let magnetization = Magnetization::new(&liou);
        let mut v = liou.product_state(&n);
        let mut prop = Propagator::new(&liou, gamma, 0.5);
        let [dx, dy, dz] = channel.site_diag();
        let f = channel.feed();
        let deriv = |r: [f64; 3]| [-p.h * r[1] + gamma * dx * r[0], p.h * r[0] + gamma * dy * r[1], gamma * dz * r[2] + gamma * f];
        let mut bloch = n.clone();
        let mut worst = 0.0f64;
        for _ in 0..6 {
            prop.step(&mut v);
            for r in bloch.iter_mut() {
                let h = 1e-3;
                for _ in 0..500 {
                    let k1 = deriv(*r);
                    let k2 = deriv(std::array::from_fn(|a| r[a] + h / 2.0 * k1[a]));
                    let k3 = deriv(std::array::from_fn(|a| r[a] + h / 2.0 * k2[a]));
                    let k4 = deriv(std::array::from_fn(|a| r[a] + h * k3[a]));
                    *r = std::array::from_fn(|a| r[a] + h / 6.0 * (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a]));
                }
            }
            let site = magnetization.per_site(&v);
            for j in 0..p.l {
                for a in 0..3 {
                    worst = worst.max((site[a][j] - bloch[j][a]).abs());
                }
            }
        }
        assert!(worst < 1e-10, "{channel:?}: deviation {worst:e}");
    }
}

/// Diffusive trajectories with σˣ dephasing and depolarizing noise against the exact ρ (full H, random c), within
/// the statistical error.
#[test]
fn diffusive_channels_match_exact() {
    let p = Params::random_c(4, &mut Rng::new(61));
    let ham = Hamiltonian::new(&p);
    let bloch = reference_bloch(p.l, p.h, 0.0, None);
    let (gamma, dt, n_traj) = (0.2, 0.02, 3000);
    for channel in [Channel::X, Channel::Depolarizing] {
        let mut liou = Liouvillian::new(&p);
        liou.set_channel(channel);
        let magnetization = Magnetization::new(&liou);
        let mut v = liou.initial_state();
        let mut prop = Propagator::new(&liou, gamma, 0.5);
        let mut trajs: Vec<Trajectory> = (0..n_traj)
            .map(|k| Trajectory::product(&ham, &bloch, 7000 + k as u64).with_jumps(channel.hermitian_jumps().unwrap()))
            .collect();
        for _ in 0..4 {
            prop.step(&mut v);
            for traj in trajs.iter_mut() {
                for _ in 0..25 {
                    traj.step(gamma, dt);
                }
            }
            let exact = magnetization.measure(&v);
            for a in 0..3 {
                let x: Vec<f64> = trajs.iter().map(|t| t.magnetization()[a]).collect();
                let mean = x.iter().sum::<f64>() / n_traj as f64;
                let sd = (x.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n_traj - 1) as f64).sqrt();
                let err = sd / (n_traj as f64).sqrt();
                assert!((mean - exact[a]).abs() < 5.0 * err + 2e-3, "{channel:?}, axis {a}: {mean} vs {} (σ = {err:.1e})", exact[a]);
            }
        }
    }
}
