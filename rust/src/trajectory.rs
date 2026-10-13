//! Pure-state trajectories for the dephasing Lindbladian.
//!
//! The jump operators √γ σᶻ_j are Hermitian, so the Lindblad equation can be unravelled as a
//! diffusion driven by real white noise that keeps the state normalized: every spin feels a
//! fluctuating z field,
//!
//!   d|ψ⟩ = −iH|ψ⟩ dt − i√γ Σ_j σᶻ_j |ψ⟩ dW_j − (γL/2)|ψ⟩ dt        (Itô, ⟨dW_j dW_k⟩ = δ_jk dt),
//!
//! and the average of |ψ⟩⟨ψ| over the noise obeys dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ).
//! One time step is  e^{−iH dt/2} · Π_j e^{−i√γ ΔW_j σᶻ_j} · e^{−iH dt/2}  with ΔW_j ~ N(0, dt):
//! the noise factor averages to the exact dephasing channel over dt, and the symmetric splitting
//! leaves an error O(dt²) in the averaged dynamics.
//!
//! States are vectors of 2^L amplitudes over bit strings, bit j = 0 for |↑⟩ and 1 for |↓⟩.

use crate::model::{hamiltonian_terms, Params};

/// Complex amplitude [re, im].
pub type Amp = [f64; 2];

struct Term {
    x: usize,
    z: usize,
    coef: Amp,
}

/// H as a sum of Pauli strings acting on state vectors.
pub struct Hamiltonian {
    pub l: usize,
    terms: Vec<Term>,
    /// Estimate of the spectral radius of H, used to size the Taylor steps.
    norm: f64,
}

impl Hamiltonian {
    pub fn new(params: &Params) -> Self {
        // P|s⟩ = i^{n_y} (−1)^{|z & s|} |s ^ x⟩: fold i^{n_y} into the coefficient
        let terms = hamiltonian_terms(params)
            .into_iter()
            .map(|(p, h)| {
                let coef = [[h, 0.0], [0.0, h], [-h, 0.0], [0.0, -h]][(p.n_y() % 4) as usize];
                Term { x: p.x as usize, z: p.z as usize, coef }
            })
            .collect();
        let mut ham = Hamiltonian { l: params.l, terms, norm: 0.0 };

        // power iteration from a fixed pseudo-random vector
        let dim = 1usize << params.l;
        let mut rng = Rng::new(0x5eed);
        let mut v: Vec<Amp> = (0..dim).map(|_| [rng.gaussian(), rng.gaussian()]).collect();
        let mut w = vec![[0.0; 2]; dim];
        for _ in 0..40 {
            let scale = 1.0 / norm_sqr(&v).sqrt();
            v.iter_mut().for_each(|a| *a = [a[0] * scale, a[1] * scale]);
            ham.apply(&v, &mut w);
            ham.norm = norm_sqr(&w).sqrt();
            std::mem::swap(&mut v, &mut w);
        }
        ham
    }

    /// out = H psi.
    pub fn apply(&self, psi: &[Amp], out: &mut [Amp]) {
        for (t, o) in out.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            for term in &self.terms {
                let s = t ^ term.x;
                let [pr, pi] = psi[s];
                let (cr, ci) = if (term.z & s).count_ones() & 1 == 0 {
                    (term.coef[0], term.coef[1])
                } else {
                    (-term.coef[0], -term.coef[1])
                };
                re += cr * pr - ci * pi;
                im += cr * pi + ci * pr;
            }
            *o = [re, im];
        }
    }
}

fn norm_sqr(v: &[Amp]) -> f64 {
    v.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum()
}

/// One trajectory: the state and the workspace of the Taylor series.
pub struct Trajectory<'a> {
    ham: &'a Hamiltonian,
    pub psi: Vec<Amp>,
    term: Vec<Amp>,
    tmp: Vec<Amp>,
    rng: Rng,
}

impl<'a> Trajectory<'a> {
    /// Starts in |−⟩^⊗L. Different seeds give independent noise.
    pub fn new(ham: &'a Hamiltonian, seed: u64) -> Self {
        let dim = 1usize << ham.l;
        let amp = 0.5f64.powf(ham.l as f64 / 2.0);
        let psi = (0..dim).map(|s| [if s.count_ones() % 2 == 0 { amp } else { -amp }, 0.0]).collect();
        Trajectory { ham, psi, term: vec![[0.0; 2]; dim], tmp: vec![[0.0; 2]; dim], rng: Rng::new(seed) }
    }

    /// psi ← e^{−iH tau} psi, Taylor series summed to round-off in substeps of norm ≤ 1.
    fn unitary(&mut self, tau: f64) {
        let n_sub = (1.5 * self.ham.norm * tau).ceil().max(1.0) as usize;
        let tau = tau / n_sub as f64;
        for _ in 0..n_sub {
            self.term.copy_from_slice(&self.psi);
            for order in 1..=60 {
                self.ham.apply(&self.term, &mut self.tmp);
                // term ← (−i tau / order) H term
                let f = tau / order as f64;
                let mut size = 0.0;
                for ((t, h), p) in self.term.iter_mut().zip(&self.tmp).zip(self.psi.iter_mut()) {
                    *t = [h[1] * f, -h[0] * f];
                    p[0] += t[0];
                    p[1] += t[1];
                    size += t[0] * t[0] + t[1] * t[1];
                }
                if size <= 1e-34 {
                    break;
                }
                assert!(order < 60, "Taylor series did not converge");
            }
        }
    }

    /// psi ← Π_j e^{−iθ_j σᶻ_j} psi with θ_j = √γ ΔW_j, up to a global phase.
    fn kick(&mut self, gamma: f64, dt: f64) {
        for j in 0..self.ham.l {
            let theta = (gamma * dt).sqrt() * self.rng.gaussian();
            // relative phase e^{2iθ} on |↓⟩_j
            let (sin, cos) = (2.0 * theta).sin_cos();
            let bit = 1usize << j;
            for (s, a) in self.psi.iter_mut().enumerate() {
                if s & bit != 0 {
                    *a = [a[0] * cos - a[1] * sin, a[0] * sin + a[1] * cos];
                }
            }
        }
    }

    /// Advance by dt.
    pub fn step(&mut self, gamma: f64, dt: f64) {
        self.unitary(dt / 2.0);
        if gamma > 0.0 {
            self.kick(gamma, dt);
        }
        self.unitary(dt / 2.0);
    }

    /// [m_x, m_y, m_z], m_α = (1/L) Σ_j ⟨ψ|σ^α_j|ψ⟩.
    pub fn magnetization(&self) -> [f64; 3] {
        let l = self.ham.l;
        let mut m = [0.0; 3];
        for j in 0..l {
            let bit = 1usize << j;
            for s in (0..self.psi.len()).filter(|s| s & bit == 0) {
                let (up, down) = (self.psi[s], self.psi[s | bit]);
                // conj(down) · up
                let (re, im) = (down[0] * up[0] + down[1] * up[1], down[0] * up[1] - down[1] * up[0]);
                m[0] += 2.0 * re;
                m[1] -= 2.0 * im;
                m[2] += up[0] * up[0] + up[1] * up[1] - down[0] * down[0] - down[1] * down[1];
            }
        }
        m.map(|x| x / l as f64)
    }
}

/// xoshiro256++ with Gaussian samples (Box-Muller).
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 expands the seed into the state
        let mut z = seed;
        let s = [0; 4].map(|_| {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        });
        Rng { s, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal.
    pub fn gaussian(&mut self) -> f64 {
        if let Some(x) = self.spare.take() {
            return x;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let (sin, cos) = (2.0 * std::f64::consts::PI * self.uniform()).sin_cos();
        self.spare = Some(r * sin);
        r * cos
    }
}
