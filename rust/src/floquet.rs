//! Trotterized (Floquet) evolution: one step is
//!
//! ```text
//! F = e^{𝓗_1 dt} e^{𝓗_2 dt} ⋯ e^{𝓗_n dt} e^{(𝓩 + 𝓓) dt},      ρ(N dt) = F^N ρ(0),
//! ```
//!
//! 𝓗_ℓ = −i[H_ℓ, ·], 𝓩 = −i[h Sᶻ_tot, ·], 𝓓 the single-site noise; the rightmost factor acts first.
//!
//! - Blocks: H_I = Σ_k h_k, h_k the term of bond (k+1, k+2) (`model::bond_terms`), on sites k … k+3 (mod L). Each
//!   h_k annihilates the tower on its own.
//! - Layers: H_ℓ = Σ_{k ≡ ℓ−1 (mod n)} h_k. The blocks of one layer are disjoint, so e^{𝓗_ℓ dt} is exact: conjugation
//!   by Π_k e^{−i h_k dt}. n = 4 when 4 divides L; otherwise no 4 layers of disjoint blocks exist on the ring, and n
//!   is the smallest number for which the blocks k ≡ ℓ−1 (mod n) are disjoint (L = 6: 6, 10: 5, 14: 5). The Trotter
//!   error depends on the layering, so a size scan should use L divisible by 4 only (the same F at every L).
//! - Noise: a Hermitian Pauli channel with equal σˣ and σʸ rates (σᶻ dephasing, dep, depol, xyz; noise.rs). Then 𝓓
//!   commutes with 𝓩 (it is covariant under rotations about z), both act site by site, and e^{(𝓩 + 𝓓)dt} is exact:
//!   on each site a rotation by h dt about z and the damping e^{d_α γ dt} of the factor σ^α (d_α from
//!   `Channel::site_diag`).
//! - Only the splitting is approximate: error O(dt²) per step from the commutators of the layers with each other and
//!   with 𝓓. At γ = 0 the scar state stays exact for any dt (every gate leaves the tower unchanged).
//!
//! The single-site factor of site s is folded into the first gate applied that contains s (layer n first, then
//! n − 1, …); it commutes with every gate applied before. One step is then L gates, with the unitary part
//! w_k = e^{−i h_k dt} e^{−i (h/2) dt Σ_{s∈S_k} σᶻ_s} (S_k the sites folded into block k).
//!
//! - ρ (`Floquet`), full Pauli basis (r_P = Tr ρP, index x | z << L as `Sector::full`): gate k is the real 256 × 256
//!   matrix G_k[P', P] = Tr(P' w_k P w_k†)/16 · Π_{s∈S_k} e^{d(P_s) γ dt} on the Pauli labels of sites k … k+3. F itself
//!   is never formed (dense, 4^L × 4^L). Cost per step: L · 256 · 4^L multiply-adds; memory 8 B · 4^L (L = 16: 34 GB).
//! - Trajectories (`FloquetTraj`), state vector ψ: gate k is the 16 × 16 unitary w_k · Π_{s∈S_k} κ_s with κ_s a random
//!   rotation e^{−iθ_x σˣ_s} e^{−iθ_y σʸ_s} e^{−iθ_z σᶻ_s}, θ_α ~ N(0, r_α γ dt) (r_α the rate of the jump √(r_α γ) σ^α).
//!   Averaged, e^{−iθσ^α} ρ e^{iθσ^α} damps the two Pauli factors that anticommute with σ^α by E[cos 2θ] = e^{−2 r_α γ dt},
//!   which is e^{𝓓 dt} of that jump exactly; the three averaged maps are diagonal in the Pauli basis and commute. So
//!   the trajectory average is F^N ρ(0) for any dt, up to statistical error only. Cost per step and trajectory:
//!   L · 16 complex multiply-adds per amplitude (64 L 2^L real), memory 16 B · 2^L.

use rayon::prelude::*;

use crate::model::{bond_terms, Params};
use crate::noise::Channel;
use crate::pauli::Pauli;
use crate::trajectory::{Amp, Rng};

type C = [f64; 2];
type M16 = [[C; 16]; 16];

/// Bases gathered together in one gate application (the inner, vectorized dimension).
const BATCH: usize = 64;

/// Trotter step (= output step of the binaries) at γ > 0 for a requested dt: dt up to γ = 0.5, then dt · 0.5/γ, so that
/// the decay (rate ≥ 2γ) stays resolved; halving dt halves it at every γ.
pub fn step_at(dt: f64, gamma: f64) -> f64 {
    dt * (0.5 / gamma).min(1.0)
}

/// Number of layers: 4 if 4 divides L, otherwise the smallest n for which the blocks k ≡ ℓ (mod n) are disjoint.
pub fn layer_count(l: usize) -> usize {
    let dist = |a: usize, b: usize| (a + l - b) % l;
    (4..=l)
        .find(|&n| (0..l).all(|k| (k + 1..l).filter(|&k2| k2 % n == k % n).all(|k2| dist(k2, k).min(dist(k, k2)) >= 4)))
        .unwrap()
}

/// One block of the schedule: sites k … k+3, which of them carry the single-site factor, and w = e^{−i h_k dt} ·
/// e^{−i (h/2) dt Σ_{folded} σᶻ} on the local qubits (qubit q = sites[q]).
struct Block {
    sites: [usize; 4],
    fold: [bool; 4],
    w: M16,
}

/// The blocks in the order of application (layer n first) and the number of layers.
fn schedule(params: &Params, dt: f64) -> (usize, Vec<Block>) {
    let l = params.l;
    assert!(params.periodic, "floquet: periodic chain only");
    let layers = layer_count(l);
    let order: Vec<usize> = (0..layers).rev().flat_map(|r| (0..l).filter(move |k| k % layers == r)).collect();
    let mut folded = vec![false; l];
    let blocks = order
        .into_iter()
        .map(|k| {
            let sites: [usize; 4] = std::array::from_fn(|q| (k + q) % l);
            let fold: [bool; 4] = std::array::from_fn(|q| !std::mem::replace(&mut folded[sites[q]], true));
            let w = block_unitary(params, k, sites, fold, dt);
            Block { sites, fold, w }
        })
        .collect();
    assert!(folded.iter().all(|&f| f));
    (layers, blocks)
}

/// w = e^{−i h_k dt} · e^{−i (h/2) dt Σ_{q folded} σᶻ_q} on the 4 local qubits.
fn block_unitary(params: &Params, k: usize, sites: [usize; 4], fold: [bool; 4], dt: f64) -> M16 {
    let l = params.l;
    let mut h = [[[0.0; 2]; 16]; 16];
    let support: u32 = sites.iter().map(|&s| 1u32 << s).sum();
    for (p, coef) in bond_terms(params, (k + 1) % l) {
        assert_eq!((p.x | p.z) & !support, 0, "bond term outside its block");
        let mut local = Pauli::IDENTITY;
        for (q, &s) in sites.iter().enumerate() {
            local.x |= ((p.x >> s) & 1) << q;
            local.z |= ((p.z >> s) & 1) << q;
        }
        for s in 0..16u32 {
            let (t, phase) = local.apply(s);
            let v = power_of_i(phase);
            let e = &mut h[t as usize][s as usize];
            e[0] += coef * v[0];
            e[1] += coef * v[1];
        }
    }
    let a: M16 = h.map(|row| row.map(|z| [z[1] * dt, -z[0] * dt])); // −i h dt
    let mut w = expm16(&a);
    for b in 0..16 {
        let field: f64 = (0..4).filter(|&q| fold[q]).map(|q| if (b >> q) & 1 == 0 { 1.0 } else { -1.0 }).sum();
        let (sin, cos) = (-params.h / 2.0 * dt * field).sin_cos();
        for row in w.iter_mut() {
            row[b] = cmul(row[b], [cos, sin]);
        }
    }
    w
}

/// The noise channels the splitting is written for: Hermitian Pauli channels with equal σˣ and σʸ rates.
fn check_channel(channel: &Channel) {
    let d = channel.site_diag();
    assert!(
        channel.hermitian_jumps().is_some() && channel.feed() == 0.0 && d[0] == d[1],
        "floquet: the noise must commute with the field (σᶻ, dep, depol or xyz), got {channel:?}"
    );
}

/// Index offsets of the N local labels and the free bit positions (ascending) of a gate on `sites`: `groups` = 1 for
/// state vectors (local bit q → bit sites[q]), 2 for Pauli strings (x bits 0–3 → bits sites[q], z bits 4–7 → bits
/// L + sites[q]).
fn layout<const N: usize>(sites: [usize; 4], l: usize, groups: usize) -> ([usize; N], Vec<u32>) {
    let mut offsets = [0usize; N];
    for (label, off) in offsets.iter_mut().enumerate() {
        for g in 0..groups {
            for (q, &s) in sites.iter().enumerate() {
                *off |= ((label >> (4 * g + q)) & 1) << (g * l + s);
            }
        }
    }
    let touched = offsets[N - 1];
    let free = (0..(groups * l) as u32).filter(|&b| (touched >> b) & 1 == 0).collect();
    (offsets, free)
}

// ---------------------------------------------------------------------------------------------------------------
// ρ in the Pauli basis

/// One 4-site gate on ρ: G (row-major, 256 × 256) on the Pauli labels of its sites.
struct Gate {
    /// Global index offset of each local label (x bits 0–3, z bits 4–7 of the label).
    offsets: [usize; 256],
    /// Bit positions of the global index that the gate does not touch, ascending.
    free: Vec<u32>,
    g: Vec<f64>,
}

pub struct Floquet {
    pub l: usize,
    /// Number of layers n.
    pub layers: usize,
    /// In the order of application.
    gates: Vec<Gate>,
}

impl Floquet {
    /// The gates of F for time step `dt` (periodic chain).
    pub fn new(params: &Params, channel: &Channel, gamma: f64, dt: f64) -> Self {
        let l = params.l;
        assert!((4..=16).contains(&l), "floquet: 4 ≤ L ≤ 16 (4^L coordinates)");
        check_channel(channel);
        let d = channel.site_diag();
        let (layers, blocks) = schedule(params, dt);
        let gates = blocks
            .iter()
            .map(|b| {
                let (offsets, free) = layout::<256>(b.sites, l, 2);
                Gate { offsets, free, g: pauli_gate(&b.w, b.fold, d, gamma, dt) }
            })
            .collect();
        Floquet { l, layers, gates }
    }

    /// r → F r.
    pub fn step(&self, r: &mut [f64]) {
        assert_eq!(r.len(), 1usize << (2 * self.l));
        for gate in &self.gates {
            gate.apply(r);
        }
    }
}

/// G[P'][P] = Re Tr(P' w P w†) / 16 · damping(P), damping = Π over the folded qubits of e^{d_α γ dt} for the factor
/// σ^α of P there.
fn pauli_gate(w: &M16, fold: [bool; 4], d: [f64; 3], gamma: f64, dt: f64) -> Vec<f64> {
    let mut g = vec![0.0; 256 * 256];
    for label in 0..256u32 {
        let p = Pauli { x: label & 15, z: label >> 4 };
        // P w†: (P w†)[c ⊕ x][b] = i^{k(c)} conj(w[b][c])
        let mut pw = [[[0.0; 2]; 16]; 16];
        for c in 0..16u32 {
            let (t, phase) = p.apply(c);
            let v = power_of_i(phase);
            for b in 0..16 {
                let wbc = w[b][c as usize];
                pw[t as usize][b] = cmul(v, [wbc[0], -wbc[1]]);
            }
        }
        let m = matmul16(w, &pw);
        let rate: f64 = (0..4)
            .filter(|&q| fold[q])
            .map(|q| match ((p.x >> q) & 1, (p.z >> q) & 1) {
                (0, 0) => 0.0,
                (1, 0) => d[0],
                (1, 1) => d[1],
                _ => d[2],
            })
            .sum();
        let damping = (rate * gamma * dt).exp();
        for label2 in 0..256u32 {
            let p2 = Pauli { x: label2 & 15, z: label2 >> 4 };
            // Tr(P' M) = Σ_u i^{k'(u)} M[u][u ⊕ x']
            let mut tr = [0.0; 2];
            for u0 in 0..16u32 {
                let (t, phase) = p2.apply(u0);
                let z = cmul(power_of_i(phase), m[u0 as usize][t as usize]);
                tr[0] += z[0];
                tr[1] += z[1];
            }
            assert!(tr[1].abs() < 1e-10, "Tr(P' w P w†) not real: {}", tr[1]);
            g[label2 as usize * 256 + label as usize] = tr[0] / 16.0 * damping;
        }
    }
    g
}

impl Gate {
    fn apply(&self, r: &mut [f64]) {
        let n_bases = 1usize << self.free.len();
        let batch = BATCH.min(n_bases);
        let ptr = SharedPtr(r.as_mut_ptr());
        (0..n_bases / batch).into_par_iter().for_each_init(
            || (vec![0.0; 256 * batch], vec![0.0; 256 * batch], vec![0usize; batch]),
            |(input, output, bases), chunk| {
                for (i, base) in bases.iter_mut().enumerate() {
                    *base = deposit(chunk * batch + i, &self.free);
                }
                // every (base, label) index belongs to exactly one chunk: no two threads touch the same entry
                let r = ptr.get();
                for (label, &off) in self.offsets.iter().enumerate() {
                    for (i, &base) in bases.iter().enumerate() {
                        input[label * batch + i] = unsafe { *r.add(base | off) };
                    }
                }
                output.fill(0.0);
                for (row, out) in self.g.chunks_exact(256).zip(output.chunks_exact_mut(batch)) {
                    for (&coef, inp) in row.iter().zip(input.chunks_exact(batch)) {
                        if coef != 0.0 {
                            for (o, &x) in out.iter_mut().zip(inp) {
                                *o += coef * x;
                            }
                        }
                    }
                }
                for (label, &off) in self.offsets.iter().enumerate() {
                    for (i, &base) in bases.iter().enumerate() {
                        unsafe { *r.add(base | off) = output[label * batch + i] };
                    }
                }
            },
        );
    }
}

struct SharedPtr(*mut f64);
unsafe impl Sync for SharedPtr {}
impl SharedPtr {
    fn get(&self) -> *mut f64 {
        self.0
    }
}

/// r_P = Π_j n_j^{a_j} of ⊗_j (1 + n_j·σ_j)/2 in the full basis (as `Liouvillian::product_state`).
pub fn product_state(bloch: &[[f64; 3]]) -> Vec<f64> {
    let l = bloch.len();
    let mask = (1usize << l) - 1;
    (0..1usize << (2 * l))
        .into_par_iter()
        .map(|code| {
            let (x, z) = (code & mask, code >> l);
            (0..l)
                .map(|j| match ((x >> j) & 1, (z >> j) & 1) {
                    (0, 0) => 1.0,
                    (1, 0) => bloch[j][0],
                    (1, 1) => bloch[j][1],
                    _ => bloch[j][2],
                })
                .product()
        })
        .collect()
}

/// [⟨σˣ_j⟩, ⟨σʸ_j⟩, ⟨σᶻ_j⟩] for j = 0 … L − 1, read off the full-basis vector.
pub fn per_site(r: &[f64], l: usize) -> [Vec<f64>; 3] {
    [0, 1, 2].map(|axis| {
        (0..l)
            .map(|j| {
                let p = Pauli::single(axis, j);
                r[p.x as usize | (p.z as usize) << l]
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------------------------------------------
// trajectories

/// One 4-site gate on ψ.
struct PsiGate {
    fold: [bool; 4],
    w: M16,
    offsets: [usize; 16],
    free: Vec<u32>,
}

pub struct FloquetTraj {
    pub l: usize,
    pub layers: usize,
    gates: Vec<PsiGate>,
    /// (axis, standard deviation of θ) of the random rotations of every site and step.
    kicks: Vec<(usize, f64)>,
}

impl FloquetTraj {
    pub fn new(params: &Params, channel: &Channel, gamma: f64, dt: f64) -> Self {
        let l = params.l;
        assert!((4..=30).contains(&l), "floquet trajectories: 4 ≤ L ≤ 30");
        check_channel(channel);
        let (layers, blocks) = schedule(params, dt);
        let gates = blocks
            .into_iter()
            .map(|b| {
                let (offsets, free) = layout::<16>(b.sites, l, 1);
                PsiGate { fold: b.fold, w: b.w, offsets, free }
            })
            .collect();
        let kicks = if gamma > 0.0 {
            channel.hermitian_jumps().unwrap().into_iter().map(|(axis, rate)| (axis, (rate * gamma * dt).sqrt())).collect()
        } else {
            Vec::new()
        };
        FloquetTraj { l, layers, gates, kicks }
    }

    /// One step of one trajectory: ψ → (Π gates, each with its random rotations) ψ.
    pub fn step(&self, psi: &mut [Amp], rng: &mut Rng) {
        assert_eq!(psi.len(), 1usize << self.l);
        for gate in &self.gates {
            let mut w = gate.w;
            for q in (0..4).filter(|&q| gate.fold[q]) {
                for &(axis, sd) in &self.kicks {
                    let (sin, cos) = (sd * rng.gaussian()).sin_cos();
                    // e^{−iθσ^α} on (|↑⟩, |↓⟩)
                    let k: [[C; 2]; 2] = match axis {
                        0 => [[[cos, 0.0], [0.0, -sin]], [[0.0, -sin], [cos, 0.0]]],
                        1 => [[[cos, 0.0], [-sin, 0.0]], [[sin, 0.0], [cos, 0.0]]],
                        _ => [[[cos, -sin], [0.0, 0.0]], [[0.0, 0.0], [cos, sin]]],
                    };
                    right_multiply_qubit(&mut w, q, &k);
                }
            }
            apply16(&w, &gate.offsets, &gate.free, psi);
        }
    }
}

/// w ← w · (k on qubit q).
fn right_multiply_qubit(w: &mut M16, q: usize, k: &[[C; 2]; 2]) {
    let bit = 1usize << q;
    for row in w.iter_mut() {
        for b0 in (0..16).filter(|b| b & bit == 0) {
            let (c0, c1) = (row[b0], row[b0 | bit]);
            // new[b] = Σ_c w[c] k[c_q][b_q]
            let n0 = cadd(cmul(c0, k[0][0]), cmul(c1, k[1][0]));
            let n1 = cadd(cmul(c0, k[0][1]), cmul(c1, k[1][1]));
            row[b0] = n0;
            row[b0 | bit] = n1;
        }
    }
}

/// ψ ← W ψ for a 16 × 16 W on the qubits of `offsets` (sequential; the parallelism is over trajectories).
fn apply16(w: &M16, offsets: &[usize; 16], free: &[u32], psi: &mut [Amp]) {
    let n_bases = 1usize << free.len();
    let batch = BATCH.min(n_bases);
    let (wr, wi): (Vec<f64>, Vec<f64>) = w.iter().flatten().map(|z| (z[0], z[1])).unzip();
    let mut buf = vec![0.0; 4 * 16 * batch];
    let (in_re, rest) = buf.split_at_mut(16 * batch);
    let (in_im, rest) = rest.split_at_mut(16 * batch);
    let (out_re, out_im) = rest.split_at_mut(16 * batch);
    let mut bases = vec![0usize; batch];
    for chunk in 0..n_bases / batch {
        for (i, base) in bases.iter_mut().enumerate() {
            *base = deposit(chunk * batch + i, free);
        }
        for (a, &off) in offsets.iter().enumerate() {
            for (i, &base) in bases.iter().enumerate() {
                let z = psi[base | off];
                in_re[a * batch + i] = z[0];
                in_im[a * batch + i] = z[1];
            }
        }
        out_re.fill(0.0);
        out_im.fill(0.0);
        for a2 in 0..16 {
            let (o_re, o_im) = (&mut out_re[a2 * batch..(a2 + 1) * batch], &mut out_im[a2 * batch..(a2 + 1) * batch]);
            for a in 0..16 {
                let (cr, ci) = (wr[a2 * 16 + a], wi[a2 * 16 + a]);
                let (x_re, x_im) = (&in_re[a * batch..(a + 1) * batch], &in_im[a * batch..(a + 1) * batch]);
                for i in 0..batch {
                    o_re[i] += cr * x_re[i] - ci * x_im[i];
                    o_im[i] += cr * x_im[i] + ci * x_re[i];
                }
            }
        }
        for (a, &off) in offsets.iter().enumerate() {
            for (i, &base) in bases.iter().enumerate() {
                psi[base | off] = [out_re[a * batch + i], out_im[a * batch + i]];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// small complex helpers

/// Spreads the bits of `b` over the positions `free` (lowest bit of b to free[0]).
fn deposit(b: usize, free: &[u32]) -> usize {
    free.iter().enumerate().fold(0, |acc, (i, &pos)| acc | ((b >> i) & 1) << pos)
}

fn power_of_i(k: u32) -> C {
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]][(k % 4) as usize]
}

fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn cadd(a: C, b: C) -> C {
    [a[0] + b[0], a[1] + b[1]]
}

fn matmul16(a: &M16, b: &M16) -> M16 {
    let mut c = [[[0.0; 2]; 16]; 16];
    for i in 0..16 {
        for k in 0..16 {
            let aik = a[i][k];
            for j in 0..16 {
                c[i][j] = cadd(c[i][j], cmul(aik, b[k][j]));
            }
        }
    }
    c
}

/// e^A for a 16 × 16 complex matrix: scaling to norm ≤ 1/2, Taylor series to round-off, squaring.
fn expm16(a: &M16) -> M16 {
    let norm = a.iter().map(|row| row.iter().map(|z| z[0].hypot(z[1])).sum::<f64>()).fold(0.0, f64::max);
    let squarings = if norm > 0.5 { (norm / 0.5).log2().ceil() as u32 } else { 0 };
    let scale = 0.5f64.powi(squarings as i32);
    let a: M16 = a.map(|row| row.map(|z| [z[0] * scale, z[1] * scale]));
    let mut e = [[[0.0; 2]; 16]; 16];
    let mut term = [[[0.0; 2]; 16]; 16];
    for i in 0..16 {
        e[i][i] = [1.0, 0.0];
        term[i][i] = [1.0, 0.0];
    }
    for order in 1..=40 {
        term = matmul16(&term, &a);
        let f = 1.0 / order as f64;
        let mut size = 0.0f64;
        for i in 0..16 {
            for j in 0..16 {
                term[i][j] = [term[i][j][0] * f, term[i][j][1] * f];
                e[i][j] = cadd(e[i][j], term[i][j]);
                size = size.max(term[i][j][0].abs()).max(term[i][j][1].abs());
            }
        }
        if size < 1e-18 {
            break;
        }
    }
    for _ in 0..squarings {
        e = matmul16(&e, &e);
    }
    e
}
