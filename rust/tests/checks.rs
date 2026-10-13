//! The two exact references of CLAUDE.md. Run with `cargo test --release`.

use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::Params;

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
