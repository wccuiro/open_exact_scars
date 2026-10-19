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
