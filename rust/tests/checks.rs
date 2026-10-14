//! The two exact references of CLAUDE.md. Run with `cargo test --release`.

use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator, Sector};
use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{random_direction, Rng};

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
