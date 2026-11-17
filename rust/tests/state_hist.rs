//! Checks of the decay-rate histogram code (`state_hist`, `state_hist_ed`): depolarizing noise `--noise depol`,
//! coherent and bulk-product initial states, a(t), the tower weight. Run with `cargo test --release`.

use open_exact_scars::initial::{bulk_product, coherent_bloch, deviation, pauli_coordinates};
use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator, Sector};
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::tower;
use open_exact_scars::trajectory::{per_site_expectations, product_amplitudes, random_direction, Amp, Hamiltonian, Rng, Trajectory};

fn full_liouvillian(p: &Params, channel: Channel) -> Liouvillian {
    let mut liou = Liouvillian::with_sector(p, Sector::full(p.l));
    liou.set_channel(channel);
    liou
}

/// Depolarizing noise with H_I = 0: every spin precesses about z and its Bloch vector shrinks as e^{−4γt/3};
/// Tr ρ stays 1; ρ → 𝟙/2^L (also with the full H).
#[test]
fn depol_free_spins() {
    let p = Params::free_spins(4);
    let mut rng = Rng::new(71);
    let n: Vec<[f64; 3]> = (0..p.l).map(|_| random_direction(&mut rng)).collect();
    let liou = full_liouvillian(&p, Channel::Depol);
    let magnetization = Magnetization::new(&liou);
    let (gamma, dt) = (0.3, 0.1);
    let mut v = liou.product_state(&n);
    let mut prop = Propagator::new(&liou, gamma, dt);
    let (mut worst, mut worst_trace) = (0.0f64, 0.0f64);
    for step in 1..=400 {
        prop.step(&mut v);
        let t = step as f64 * dt;
        let decay = (-4.0 * gamma * t / 3.0).exp();
        let (sin, cos) = (p.h * t).sin_cos();
        let site = magnetization.per_site(&v);
        for j in 0..p.l {
            let ex = [(n[j][0] * cos - n[j][1] * sin) * decay, (n[j][0] * sin + n[j][1] * cos) * decay, n[j][2] * decay];
            for a in 0..3 {
                worst = worst.max((site[a][j] - ex[a]).abs());
            }
        }
        worst_trace = worst_trace.max((v[0] - 1.0).abs());
    }
    assert!(worst < 1e-11, "Bloch vectors: deviation {worst:e}");
    assert!(worst_trace < 1e-12, "Tr ρ − 1 = {worst_trace:e}");
    // t = 40: e^{−4γt/3} = e^{−16}
    let rest = v[1..].iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(rest < 1e-6, "free spins: largest r_P (P ≠ 𝟙) at t = 40 is {rest:e}");

    // full H: −(4γ/3) w(P) ≤ −4γ/3 on every traceless string and the H part is antisymmetric, so ρ → 𝟙/2^L as well
    let p = Params::random_c(4, &mut Rng::new(72));
    let liou = full_liouvillian(&p, Channel::Depol);
    let mut v = liou.product_state(&n);
    let mut prop = Propagator::new(&liou, gamma, 1.0);
    for _ in 0..40 {
        prop.step(&mut v);
    }
    let rest = v[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((v[0] - 1.0).abs() < 1e-12 && rest < 1e-6 * 2f64.sqrt() * 16.0, "full H: Tr ρ = {}, ‖r_traceless‖ = {rest:e}", v[0]);
}

/// The tower weight W = 2^{−L} Σ_P r_P Tr(P_tower P) starts with slope −(2γ/3)(L − 1) for every state in the tower
/// (coherent states at several θ, φ and a random superposition of the |Φ_n⟩).
#[test]
fn tower_weight_initial_slope() {
    let p = Params::random_c(6, &mut Rng::new(73));
    let l = p.l;
    let gamma = 0.2;
    let liou = full_liouvillian(&p, Channel::Depol);
    let traces = tower::traces(l, &vec![1.0; l + 1]);
    let norm = 0.5f64.powi(l as i32);
    let mut states: Vec<Vec<f64>> =
        [(0.3, 0.0), (std::f64::consts::FRAC_PI_2, std::f64::consts::PI), (2.2, 1.1)].iter().map(|&(th, ph)| liou.product_state(&coherent_bloch(l, th, ph))).collect();
    let mut rng = Rng::new(74);
    let phi = tower::states(l);
    let c: Vec<Amp> = (0..=l).map(|_| [rng.gaussian(), rng.gaussian()]).collect();
    let cn = c.iter().map(|z| z[0] * z[0] + z[1] * z[1]).sum::<f64>().sqrt();
    let psi: Vec<Amp> =
        (0..1usize << l).map(|s| (0..=l).fold([0.0, 0.0], |acc, n| [acc[0] + c[n][0] / cn * phi[n][s], acc[1] + c[n][1] / cn * phi[n][s]])).collect();
    assert!((tower::weight(&psi, l) - 1.0).abs() < 1e-13);
    states.push(pauli_coordinates(&psi, l));
    let mut out = vec![0.0; liou.dim()];
    for v in &states {
        let w: f64 = norm * v.iter().zip(&traces).map(|(r, t)| r * t).sum::<f64>();
        assert!((w - 1.0).abs() < 1e-12, "W(0) = {w}");
        liou.apply(gamma, v, &mut out);
        let slope: f64 = norm * out.iter().zip(&traces).map(|(r, t)| r * t).sum::<f64>();
        let expected = -(2.0 * gamma / 3.0) * (l - 1) as f64;
        assert!((slope - expected).abs() < 1e-11, "dW/dt(0) = {slope}, expected {expected}");
    }
}

/// Diffusive depolarizing trajectories (one isotropic rotation per site and step) against the exact ρ at L = 8,
/// 2000 trajectories: per-site ⟨σ^α_j⟩, a(t) and W(t) within one standard error (rms of the deviations in units of
/// the standard error ≤ 1.3, none beyond 4.5).
#[test]
fn diffusive_depol_matches_exact() {
    let p = Params::random_c(8, &mut Rng::new(75));
    let l = p.l;
    let ham = Hamiltonian::new(&p);
    let bloch = coherent_bloch(l, 0.4f64.acos(), std::f64::consts::PI);
    let (gamma, dt, n_traj) = (0.1, 0.1, 2000);
    let liou = full_liouvillian(&p, Channel::Depol);
    let magnetization = Magnetization::new(&liou);
    let traces = tower::traces(l, &vec![1.0; l + 1]);
    let mut v = liou.product_state(&bloch);
    let mut prop = Propagator::new(&liou, gamma, 0.5);
    let psi0 = product_amplitudes(&bloch);
    let mut trajs: Vec<Trajectory> =
        (0..n_traj).map(|k| Trajectory::from_state(&ham, psi0.clone(), 9000 + k as u64).with_channel(&Channel::Depol)).collect();
    let nbar = vec![[0.0; 2]; l];
    let nf = n_traj as f64;
    let (mut sum_sq, mut count, mut worst) = (0.0, 0, 0.0f64);
    for _ in 0..6 {
        prop.step(&mut v);
        use rayon::prelude::*;
        trajs.par_iter_mut().for_each(|t| (0..5).for_each(|_| t.step(gamma, dt)));
        let exact = magnetization.per_site(&v);
        let w_exact: f64 = 0.5f64.powi(l as i32) * v.iter().zip(&traces).map(|(r, t)| r * t).sum::<f64>();
        let per: Vec<([Vec<f64>; 3], f64)> = trajs.par_iter().map(|t| (t.per_site(), tower::weight(&t.psi, l))).collect();
        let mut z = |x: Vec<f64>, reference: f64| {
            let m = x.iter().sum::<f64>() / nf;
            let se = (x.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (nf - 1.0) / nf).sqrt();
            let d = (m - reference) / se;
            sum_sq += d * d;
            count += 1;
            worst = worst.max(d.abs());
        };
        for a in 0..3 {
            for j in 0..l {
                z(per.iter().map(|(s, _)| s[a][j]).collect(), exact[a][j]);
            }
        }
        z(per.iter().map(|(_, w)| *w).collect(), w_exact);
        // a(t) from the trajectory means, error by the delta method (as in state_hist)
        let mean: Vec<[f64; 2]> = (0..l).map(|j| [0, 1].map(|a| per.iter().map(|(s, _)| s[a][j]).sum::<f64>() / nf)).collect();
        let a_traj = deviation(&mean.iter().map(|m| m[0]).collect::<Vec<_>>(), &mean.iter().map(|m| m[1]).collect::<Vec<_>>(), &nbar);
        let a_exact = deviation(&exact[0], &exact[1], &nbar);
        let u: Vec<[f64; 2]> = mean.iter().map(|m| [m[0] / m[0].hypot(m[1]), m[1] / m[0].hypot(m[1])]).collect();
        let q: Vec<f64> = per.iter().map(|(s, _)| (0..l).map(|j| u[j][0] * s[0][j] + u[j][1] * s[1][j]).sum::<f64>() / l as f64).collect();
        let qm = q.iter().sum::<f64>() / nf;
        let se = (q.iter().map(|y| (y - qm).powi(2)).sum::<f64>() / (nf - 1.0) / nf).sqrt();
        let d = (a_traj - a_exact) / se;
        sum_sq += d * d;
        count += 1;
        worst = worst.max(d.abs());
    }
    let rms = (sum_sq / count as f64).sqrt();
    println!("diffusive depol vs exact, L = 8: rms {rms:.2}, max {worst:.2} standard errors over {count} values");
    assert!(rms < 1.3 && worst < 4.5, "rms {rms:.2}, max {worst:.2} standard errors");
}

/// Coherent states at γ = 0: |n⊥_j| = sin θ on every site at all times, so a(t) is constant (exact ρ and state
/// vector).
#[test]
fn coherent_a_constant_at_gamma_zero() {
    let p = Params::random_c(6, &mut Rng::new(76));
    let l = p.l;
    let ham = Hamiltonian::new(&p);
    let liou = full_liouvillian(&p, Channel::Depol);
    let magnetization = Magnetization::new(&liou);
    let nbar = vec![[0.0; 2]; l];
    for (c, phi) in [(0.0, std::f64::consts::PI), (0.4, std::f64::consts::PI), (-0.8, 0.7)] {
        let bloch = coherent_bloch(l, f64::acos(c), phi);
        let a0 = (1.0 - c * c as f64).sqrt();
        let mut v = liou.product_state(&bloch);
        let mut prop = Propagator::new(&liou, 0.0, 0.1);
        let mut traj = Trajectory::product(&ham, &bloch, 0);
        let mut worst = 0.0f64;
        for _ in 0..100 {
            prop.step(&mut v);
            traj.unitary(0.1);
            let s = magnetization.per_site(&v);
            let st = traj.per_site();
            worst = worst.max((deviation(&s[0], &s[1], &nbar) - a0).abs()).max((deviation(&st[0], &st[1], &nbar) - a0).abs());
        }
        assert!(worst < 1e-12, "cos θ = {c}: |a(t) − a(0)| up to {worst:e}");
    }
}

/// Bulk product states: normalized, no tower weight, |E/L| ≤ 0.25.
#[test]
fn bulk_states_have_no_tower_weight() {
    for l in [6, 8, 10] {
        let p = Params::random_c(l, &mut Rng::new(1000 + l as u64));
        let ham = Hamiltonian::new(&p);
        for seed in 1..=20 {
            let b = bulk_product(&ham, seed);
            let norm: f64 = b.psi.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum();
            let w = tower::weight(&b.psi, l);
            assert!((norm - 1.0).abs() < 1e-13 && w < 1e-14, "L = {l}, seed {seed}: norm {norm}, tower weight {w:e}");
            assert!((b.energy / l as f64).abs() <= 0.25);
        }
    }
}

/// γ = 0: the state-vector evolution equals the exact ρ (Pauli coordinates of the pure state) at L = 8, for a bulk
/// product state; the Pauli coordinates of a product state equal `Liouvillian::product_state`.
#[test]
fn pure_state_matches_exact_rho() {
    let p = Params::random_c(8, &mut Rng::new(77));
    let l = p.l;
    let ham = Hamiltonian::new(&p);
    let liou = full_liouvillian(&p, Channel::Depol);
    let magnetization = Magnetization::new(&liou);

    let mut rng = Rng::new(78);
    let bloch: Vec<[f64; 3]> = (0..l).map(|_| random_direction(&mut rng)).collect();
    let (a, b) = (pauli_coordinates(&product_amplitudes(&bloch), l), liou.product_state(&bloch));
    let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(diff < 1e-13, "Pauli coordinates of a product state: {diff:e}");

    let bulk = bulk_product(&ham, 5);
    let mut v = pauli_coordinates(&bulk.psi, l);
    let purity = v.iter().map(|r| r * r).sum::<f64>() / (1u64 << l) as f64;
    assert!((v[0] - 1.0).abs() < 1e-13 && (purity - 1.0).abs() < 1e-12, "Tr ρ = {}, Tr ρ² = {purity}", v[0]);
    let mut prop = Propagator::new(&liou, 0.0, 0.1);
    let mut traj = Trajectory::from_state(&ham, bulk.psi.clone(), 0);
    let mut worst = 0.0f64;
    for _ in 0..50 {
        prop.step(&mut v);
        traj.unitary(0.1);
        let (x, y) = (magnetization.per_site(&v), traj.per_site());
        for k in 0..3 {
            worst = x[k].iter().zip(&y[k]).map(|(s, t)| (s - t).abs()).fold(worst, f64::max);
        }
    }
    assert!(worst < 1e-10, "pure vs exact: deviation {worst:e}");
}

#[cfg(mkl)]
#[path = "../src/mkl_ed.rs"]
mod mkl_ed;

/// n̄⊥_j from ED (diagonal ensemble) equals the late-time average of a γ = 0 state-vector run at L = 8 (t = 50 … 250),
/// within the floor (rms of a(t) over the same window, a computed with the ED n̄).
#[cfg(mkl)]
#[test]
fn nbar_from_ed_matches_late_time_average() {
    let p = Params::random_c(8, &mut Rng::new(79));
    let l = p.l;
    let ham = Hamiltonian::new(&p);
    let eig = mkl_ed::Eigen::new(&p);
    for seed in [1, 2] {
        let bulk = bulk_product(&ham, seed);
        let (nbar_ed, norm, e_ed) = eig.diagonal_ensemble(&bulk.psi);
        assert!((norm - 1.0).abs() < 1e-12 && (e_ed - bulk.energy).abs() < 1e-10);
        let nbar: Vec<[f64; 2]> = nbar_ed.iter().map(|n| [n[0], n[1]]).collect();
        let mut traj = Trajectory::from_state(&ham, bulk.psi.clone(), 0);
        traj.unitary(50.0);
        let (mut avg, mut a_sq, mut count) = (vec![[0.0; 2]; l], 0.0, 0);
        for _ in 0..2000 {
            traj.unitary(0.1);
            let s: [Vec<f64>; 3] = per_site_expectations(&traj.psi, l);
            for j in 0..l {
                avg[j][0] += s[0][j];
                avg[j][1] += s[1][j];
            }
            a_sq += deviation(&s[0], &s[1], &nbar).powi(2);
            count += 1;
        }
        let floor = (a_sq / count as f64).sqrt();
        let worst = (0..l)
            .map(|j| (avg[j][0] / count as f64 - nbar[j][0]).hypot(avg[j][1] / count as f64 - nbar[j][1]))
            .fold(0.0, f64::max);
        println!("init seed {seed}: |n̄_ED − n̄_late| ≤ {worst:.2e}, floor {floor:.2e}");
        assert!(worst < floor, "init seed {seed}: |n̄_ED − n̄_late| = {worst:e} above the floor {floor:e}");
    }
}

/// The dense density-matrix evolution (dmatrix.rs, used by `state_hist --method exact`) equals the Pauli-basis
/// evolution (liouville.rs): per-site Bloch vectors, tower weight and Tr ρ, at L = 6 for depol, σᶻ and σˣ noise,
/// γ = 0.1 and 0, and at L = 8 for depol, γ = 0.1, from a coherent state and a bulk product state.
#[test]
fn density_matrix_matches_pauli_basis() {
    use open_exact_scars::dmatrix::{DensityLiouvillian, DensityPropagator};
    for l in [6, 8] {
        let p = Params::random_c(l, &mut Rng::new(80 + l as u64));
        let ham = Hamiltonian::new(&p);
        let traces = tower::traces(l, &vec![1.0; l + 1]);
        let norm = 0.5f64.powi(l as i32);
        let (channels, gammas) = if l == 6 { (vec![Channel::Depol, Channel::Z, Channel::X], vec![0.1, 0.0]) } else { (vec![Channel::Depol], vec![0.1]) };
        for &channel in &channels {
            for &gamma in &gammas {
                let pauli = full_liouvillian(&p, channel);
                let magnetization = Magnetization::new(&pauli);
                let dense = DensityLiouvillian::new(&p, channel);
                for psi in [product_amplitudes(&coherent_bloch(l, 0.6f64.acos(), 0.3)), bulk_product(&ham, 4).psi] {
                    let mut v = pauli_coordinates(&psi, l);
                    let mut rho = dense.pure(&psi);
                    let mut prop_v = Propagator::new(&pauli, gamma, 0.25);
                    let mut prop_r = DensityPropagator::new(&dense, gamma, 0.25);
                    let mut worst = 0.0f64;
                    for _ in 0..20 {
                        prop_v.step(&mut v);
                        prop_r.step(&mut rho);
                        let (x, y) = (magnetization.per_site(&v), dense.per_site(&rho));
                        for k in 0..3 {
                            worst = x[k].iter().zip(&y[k]).map(|(s, t)| (s - t).abs()).fold(worst, f64::max);
                        }
                        let w_pauli = norm * v.iter().zip(&traces).map(|(r, t)| r * t).sum::<f64>();
                        worst = worst.max((w_pauli - dense.tower_weight(&rho)).abs()).max((dense.trace(&rho) - 1.0).abs());
                    }
                    assert!(worst < 1e-11, "L = {l}, {channel:?}, γ = {gamma}: deviation {worst:e}");
                }
            }
        }
    }
}

/// Damage D(τ) = Σ_j [1 − Re c_j] of one kick σ^axis_k on the coherent state at θ, phase φ_m (as in
/// `single_flip --theta … --kick …`).
fn damage_theta(p: &Params, theta: f64, phase: f64, kick: (usize, usize), taus: &[f64]) -> Vec<(f64, Vec<Amp>)> {
    use open_exact_scars::trajectory::{coherences_theta, coherent_reference};
    let ham = Hamiltonian::new(p);
    let t0 = phase / p.h;
    let pi = std::f64::consts::PI;
    let mut traj = Trajectory::product(&ham, &coherent_reference(p.l, p.h, t0, theta, pi, Some(kick)), 0);
    let mut tau = 0.0;
    taus.iter()
        .map(|&t| {
            if t > tau {
                traj.unitary(t - tau);
                tau = t;
            }
            let c = coherences_theta(&traj.per_site(), p.h, t0 + tau, theta, pi);
            (c.iter().map(|c| 1.0 - c[0]).sum(), c)
        })
        .collect()
}

/// single_flip --theta --kick: D_zz(0) = 2; the phase average of D_xx(0) + D_yy(0) is 2; with H_I = 0,
/// (D_xx + D_yy + D_zz)/3 = 4/3 at every τ; the z kick at θ = π/2 gives the original single_flip output.
#[test]
fn single_flip_kicks_at_theta() {
    use open_exact_scars::trajectory::{coherences, reference_bloch};
    let full = Params::random_c(6, &mut Rng::new(90));
    let free = Params::free_spins(6);
    let taus = [0.0, 0.8, 2.9, 7.0];
    let phases: Vec<f64> = (0..4).map(|m| 2.0 * std::f64::consts::PI * m as f64 / 4.0).collect();
    for c in [0.0, 0.4, -0.4] {
        let theta = f64::acos(c);
        let mut xy = 0.0;
        for &ph in &phases {
            let dz = damage_theta(&full, theta, ph, (2, 2), &taus[..1])[0].0;
            assert!((dz - 2.0).abs() < 1e-12, "cos θ = {c}: D_zz(0) = {dz}");
            xy += damage_theta(&full, theta, ph, (3, 0), &taus[..1])[0].0 + damage_theta(&full, theta, ph, (3, 1), &taus[..1])[0].0;
            let sums: Vec<f64> = (0..3)
                .map(|a| damage_theta(&free, theta, ph, (1, a), &taus).iter().map(|d| d.0).collect::<Vec<_>>())
                .fold(vec![0.0; taus.len()], |acc, d| acc.iter().zip(&d).map(|(x, y)| x + y).collect());
            for s in sums {
                assert!((s / 3.0 - 4.0 / 3.0).abs() < 1e-11, "free spins, cos θ = {c}: (D_xx + D_yy + D_zz)/3 = {}", s / 3.0);
            }
        }
        let xy = xy / phases.len() as f64;
        assert!((xy - 2.0).abs() < 1e-12, "cos θ = {c}: phase average of D_xx(0) + D_yy(0) = {xy}");
    }
    // the z kick at θ = π/2, φ = π is the original single_flip run
    let ham = Hamiltonian::new(&full);
    for &ph in &phases {
        let general = damage_theta(&full, std::f64::consts::FRAC_PI_2, ph, (2, 2), &taus);
        let mut traj = Trajectory::product(&ham, &reference_bloch(full.l, full.h, ph / full.h, Some(2)), 0);
        let mut tau = 0.0;
        for (&t, (d, c)) in taus.iter().zip(&general) {
            if t > tau {
                traj.unitary(t - tau);
                tau = t;
            }
            let c0 = coherences(&traj.per_site(), full.h, ph / full.h + tau);
            let d0: f64 = c0.iter().map(|c| 1.0 - c[0]).sum();
            let worst = c.iter().zip(&c0).map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs())).fold((d - d0).abs(), f64::max);
            assert!(worst < 1e-12, "θ = π/2, z kick against the original: {worst:e}");
        }
    }
}
