//! Trotterized (Floquet) evolution: the circuit of floquet-circuit.md (notes, Sec. "Floquet circuit"). One step is
//!
//! ```text
//! F = 𝓔 ∘ 𝓑 ∘ 𝓤_1 ∘ 𝓤_2 ∘ ⋯ ∘ 𝓤_d,      𝓤_ℓ(ρ) = U_ℓ ρ U_ℓ†,  𝓑(ρ) = U_B ρ U_B†,      ρ(N dt) = F^N ρ(0),
//! ```
//!
//! the rightmost factor acts first: U_d, U_{d−1}, …, U_1, then the field U_B = e^{−i h dt Sᶻ_tot} on every site, then
//! the noise 𝓔 = e^{𝓓 dt} on every site.
//!
//! - Blocks: H_I = Σ_k h_k, h_k the term of bond (k+1, k+2) (`model::bond_terms`: RA, AR and ARA of that bond), on
//!   sites k … k+3 (mod L). Each h_k annihilates the tower on its own.
//! - Layers: d = the smallest divisor of L that is at least 4 (4 if 4 | L; L = 6: 6, 10: 5, 14: 7, 18: 6; L prime:
//!   d = L). Layer ℓ = 1 … d holds the blocks starting at k ≡ ℓ − 1 (mod d): L/d disjoint blocks, d − 4 idle sites
//!   after each. Every layer is the previous one shifted by one site; every bond is the middle bond of exactly one
//!   block per step. U_ℓ = Π_{k ∈ ℓ} e^{−i h_k dt} is exact (the blocks of a layer are disjoint).
//! - Field: one uniform layer after U_1, U_B = Π_s e^{−i (h/2) dt σᶻ_s}. It maps the tower onto itself and every
//!   e^{−i h_k dt} is the identity on the tower, so at γ = 0 the scar state evolves exactly (reference 1) for any dt.
//! - Noise: one layer after the field, 𝓔 = ⊗_s e^{𝓓_s dt}, any Hermitian Pauli channel (noise.rs: σᶻ, σˣ, σʸ, dep,
//!   depol, xyz; diagonal in the Pauli basis, d_α from `Channel::site_diag`).
//! - Error: first order, O(dt²) per step, from the commutators of the layers with each other, with the field and with
//!   𝓓 (outside the tower, and for the tower once the noise has moved weight out of it).
//!
//! Implementation: the field and the noise of site s are folded into the last gate applied that contains s (layer 1
//! first, then 2, …), applied after its interaction: they commute with every gate applied later (none touches s), so
//! the product is F exactly. One step is L gates, with the unitary part w_k = e^{−i (h/2) dt Σ_{s∈S_k} σᶻ_s} e^{−i h_k dt}
//! (S_k the sites folded into block k).
//!
//! - ρ (`Floquet`), full Pauli basis (r_P = Tr ρP, index x | z << L as `Sector::full`): gate k is the real 256 × 256
//!   matrix G_k[P', P] = Π_{s∈S_k} e^{d(P'_s) γ dt} · Tr(P' w_k P w_k†)/16 on the Pauli labels of sites k … k+3. F itself
//!   is never formed (dense, 4^L × 4^L). Cost per step: L · 256 · 4^L multiply-adds; memory 8 B · 4^L (L = 16: 34 GB).
//! - Trajectories (`FloquetTraj`), state vector ψ: gate k is the 16 × 16 unitary Π_{s∈S_k} κ_s · w_k with κ_s a random
//!   rotation e^{−iθ_x σˣ_s} e^{−iθ_y σʸ_s} e^{−iθ_z σᶻ_s}, θ_α ~ N(0, r_α γ dt) (r_α the rate of the jump √(r_α γ) σ^α).
//!   Averaged, e^{−iθσ^α} ρ e^{iθσ^α} damps the two Pauli factors that anticommute with σ^α by E[cos 2θ] = e^{−2 r_α γ dt},
//!   which is e^{𝓓 dt} of that jump exactly; the three averaged maps are diagonal in the Pauli basis and commute. So
//!   the trajectory average is F^N ρ(0) for any dt, up to statistical error only. Cost per step and trajectory:
//!   L · 16 complex multiply-adds per amplitude (64 L 2^L real), memory 16 B · 2^L.
//!
//! Before 2026-11-25 (commit f223cc3, "old protocol", data in data/non_used_data/floquet_old_protocol/): n = the
//! smallest disjoint spacing (L = 14: 5 layers, not a shift pattern), field and noise first, folded into the first gate
//! of every site. Both protocols keep the γ = 0 scar exact; they differ at O(dt) elsewhere.

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

/// Number of layers d: the smallest divisor of L that is at least 4.
pub fn layer_count(l: usize) -> usize {
    assert!(l >= 4, "floquet: L ≥ 4");
    (4..=l).find(|&d| l % d == 0).unwrap()
}

/// Block starts of layer ℓ = 1 … d (0-based sites): k ≡ ℓ − 1 (mod d).
pub fn layer_starts(l: usize, layer: usize) -> Vec<usize> {
    let d = layer_count(l);
    assert!((1..=d).contains(&layer));
    (layer - 1..l).step_by(d).collect()
}

/// One block of the schedule: sites k … k+3, which of them carry the field and the noise of the step (applied after
/// the interaction), and w = e^{−i (h/2) dt Σ_{folded} σᶻ} e^{−i h_k dt} on the local qubits (qubit q = sites[q]).
struct Block {
    sites: [usize; 4],
    fold: [bool; 4],
    w: M16,
}

/// The blocks in the order of application (layer d first, layer 1 last) and the number of layers d.
fn schedule(params: &Params, dt: f64) -> (usize, Vec<Block>) {
    let l = params.l;
    assert!(params.periodic, "floquet: periodic chain only");
    let layers = layer_count(l);
    let order: Vec<usize> = (1..=layers).rev().flat_map(|layer| layer_starts(l, layer)).collect();
    // field and noise last: those of site s go into the last gate applied that contains s
    let mut folded = vec![false; l];
    let mut fold = vec![[false; 4]; order.len()];
    for (i, &k) in order.iter().enumerate().rev() {
        fold[i] = std::array::from_fn(|q| !std::mem::replace(&mut folded[(k + q) % l], true));
    }
    assert!(folded.iter().all(|&f| f));
    let blocks = order
        .iter()
        .zip(fold)
        .map(|(&k, fold)| {
            let sites: [usize; 4] = std::array::from_fn(|q| (k + q) % l);
            Block { sites, fold, w: block_unitary(params, k, sites, fold, dt) }
        })
        .collect();
    (layers, blocks)
}

/// w = e^{−i (h/2) dt Σ_{q folded} σᶻ_q} · e^{−i h_k dt} on the 4 local qubits (the field after the interaction).
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
    // left multiplication by the diagonal field rotation: row b (output basis state) gets its phase
    for (b, row) in w.iter_mut().enumerate() {
        let field: f64 = (0..4).filter(|&q| fold[q]).map(|q| if (b >> q) & 1 == 0 { 1.0 } else { -1.0 }).sum();
        let (sin, cos) = (-params.h / 2.0 * dt * field).sin_cos();
        for z in row.iter_mut() {
            *z = cmul(*z, [cos, sin]);
        }
    }
    w
}

/// The noise channels the splitting is written for: Hermitian Pauli channels (diagonal in the Pauli basis, no feed).
fn check_channel(channel: &Channel) {
    assert!(
        channel.hermitian_jumps().is_some() && channel.feed() == 0.0,
        "floquet: the noise must be a Hermitian Pauli channel (σᶻ, σˣ, σʸ, dep, depol or xyz), got {channel:?}"
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

/// G[P'][P] = damping(P') · Re Tr(P' w P w†) / 16, damping = Π over the noisy qubits of e^{d_α γ dt} for the factor
/// σ^α of P' there (noise after the unitary w, which already contains the field).
fn pauli_gate(w: &M16, fold: [bool; 4], d: [f64; 3], gamma: f64, dt: f64) -> Vec<f64> {
    let damping: Vec<f64> = (0..256u32)
        .map(|label| {
            let rate: f64 = (0..4)
                .filter(|&q| fold[q])
                .map(|q| match ((label >> q) & 1, (label >> (4 + q)) & 1) {
                    (0, 0) => 0.0,
                    (1, 0) => d[0],
                    (1, 1) => d[1],
                    _ => d[2],
                })
                .sum();
            (rate * gamma * dt).exp()
        })
        .collect();
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
            g[label2 as usize * 256 + label as usize] = tr[0] / 16.0 * damping[label2 as usize];
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
    noise: [bool; 4],
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
                PsiGate { noise: b.fold, w: b.w, offsets, free }
            })
            .collect();
        let kicks = if gamma > 0.0 {
            channel.hermitian_jumps().unwrap().into_iter().map(|(axis, rate)| (axis, (rate * gamma * dt).sqrt())).collect()
        } else {
            Vec::new()
        };
        FloquetTraj { l, layers, gates, kicks }
    }

    /// One step of one trajectory: ψ → (Π gates, each followed by its random rotations) ψ.
    pub fn step(&self, psi: &mut [Amp], rng: &mut Rng) {
        assert_eq!(psi.len(), 1usize << self.l);
        for gate in &self.gates {
            let mut w = gate.w;
            for q in (0..4).filter(|&q| gate.noise[q]) {
                for &(axis, sd) in &self.kicks {
                    let (sin, cos) = (sd * rng.gaussian()).sin_cos();
                    // e^{−iθσ^α} on (|↑⟩, |↓⟩)
                    let k: [[C; 2]; 2] = match axis {
                        0 => [[[cos, 0.0], [0.0, -sin]], [[0.0, -sin], [cos, 0.0]]],
                        1 => [[[cos, 0.0], [-sin, 0.0]], [[sin, 0.0], [cos, 0.0]]],
                        _ => [[[cos, -sin], [0.0, 0.0]], [[0.0, 0.0], [cos, sin]]],
                    };
                    left_multiply_qubit(&mut w, q, &k);
                }
            }
            apply16(&w, &gate.offsets, &gate.free, psi);
        }
    }
}

/// w ← (k on qubit q) · w.
fn left_multiply_qubit(w: &mut M16, q: usize, k: &[[C; 2]; 2]) {
    let bit = 1usize << q;
    for r0 in (0..16).filter(|r| r & bit == 0) {
        for col in 0..16 {
            let (c0, c1) = (w[r0][col], w[r0 | bit][col]);
            // new[a] = Σ_c k[a_q][c_q] w[c]
            w[r0][col] = cadd(cmul(k[0][0], c0), cmul(k[0][1], c1));
            w[r0 | bit][col] = cadd(cmul(k[1][0], c0), cmul(k[1][1], c1));
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
