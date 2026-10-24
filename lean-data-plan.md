# Lean data plan: improving the scar data at minimum cost (25 Oct 2026)

Supersedes the earlier versions of this file. The goal is better data, not a proof of hydrodynamics. Existing data
(CLAUDE.md, state of 2026-10-20) is never rerun. Costs use `data/leonardo_benchmarks.md` and
`data/obelix_benchmarks.md`, in Leonardo node-hours (NH; 1 NH = 112 core-h).

## Open points this plan addresses (from CLAUDE.md)

1. Γ∞ at γ = 0.01–0.02 is the least converged: Γ_e still grows at L = 20 at γ = 0.01, and the L ≥ 8 and L ≥ 14 fits
   differ by 15 % and 8 %.
2. Growth of D̄(τ) at L ≥ 24 and τ > 8: this fixes Γ∞ at small γ through the single-kick formula.
3. Only one disorder realization per L.
4. New observables requested: entanglement, operator entanglement of ρ, operator size and support.
5. Other single-site noise channels.

## Cost-saving rules (apply to every new run)

| Rule | Saving |
|---|---|
| Stop at t ≈ 1.5 t_e (enough for Γ_e), not at m⊥ = 1e-3. The tail is only needed from exact runs. | 3–10× (e.g. L = 20, γ = 0.01 ran to t = 107 for t_e ≈ 9.4) |
| Diffusive, not jump, at small γ. Per-trajectory spread ≈ 0.08 (diffusive) against 0.3–0.4 (jump) at L = 12, so about 15–20× more jump trajectories for the same error; the first-kick saving of the jump runs recovers only ≈ 2×. | ≈ 10× per error bar |
| 112 trajectories (diffusive), not 224/448, wherever σ_Γ ≈ e·σ_m ≤ 2–3 % | 2–4× |
| single_flip: fewer phases and sites, set from the variance in the existing CSVs (A2) | ≈ 4× |
| L ≤ 12 exact and L ≤ 16 trajectories on obelix (≈ 0.1 node, free) | moves ≈ 10–15 NH off Leonardo |
| New observables only in the new runs | no reruns |

## A. Analysis of existing data (laptop, 0 NH; do first)

| # | What | Output |
|---|---|---|
| A1 | Overlay −(1/γ) ln m⊥ for L = 16, 18, 20 at γ ≤ 0.02; keep t < L/s, before the damage wraps around the ring | ∫D∞ and slope s from the Lindblad data; a check of the single-kick D̄ |
| A2 | From single_flip_L20/L24: spread of D(τ) over the 16 phases and over sites | how many phases and sites give ≤ 2 % error on D̄ (sets B2 and B3) |
| A3 | From the diffusive L = 16–20 runs: per-trajectory spread at t_e | trajectory count for a 2 % Γ_e |
| A4 | Predicted t_e for the new runs (from Γ_e(L = 20) and the single-kick formula) | t-max = 1.5 t_e per (L, γ) |

## B. Leonardo runs

| # | Run | Details | Estimate |
|---|---|---|---|
| B1 | Scar, diffusive, L = 22 | γ = 0.005, 0.01, 0.02, 0.05; 112 traj; t-max ≈ 24, 14, 9, 5 (1.5 t_e); Rényi-2 and half-chain S_A every Δt = 0.5 (< 1 % overhead). Wall ≈ 1500 s per unit t. | ≈ 22 NH (2 jobs) |
| B1′ | Scar, diffusive, L = 24 | γ = 0.01, 0.02; 112 traj (90 GB); ≈ 7300 s per unit t | ≈ 47 NH (3 jobs); optional, decide after B1 |
| B2 | single_flip, longer τ | L = 24 to τ = 16 and L = 26 to τ = 14, with the phases × sites from A2 (assume 4 × 4 = 16 runs); entanglement across every cut every Δτ = 1 (≈ 5 % overhead) | ≈ 4 NH (L = 24) + ≈ 17 NH (L = 26, 51 GB) |
| B3 | Disorder | 3 more realizations: single_flip L = 20 (4 phases × all sites, τ ≤ 25, ≈ 1.6 NH each); diffusive L = 20, γ = 0.01, 0.05, 0.2, 112 traj, t = 1.5 t_e (≈ 2.4 NH each); the same at L = 16 on obelix | ≈ 12 NH |
| B4 | Exact ρ with the new observables, L = 12 | γ = 0, 0.005, 0.01, 0.05, 0.2, 1; t = 1.5 t_e (γ = 0: t = 20). Output per step: purity, operator entanglement S_op(ρ) (SVD every Δt = 0.5), operator-size distribution P(n) and ⟨n_xy⟩, support distribution P(ℓ) (from r_P, one pass). L = 8 and 10 on obelix or the laptop. | ≈ 5 NH |
| B5 | Other single-site noise channels, uniform | σˣ and σʸ dephasing, depolarizing (diagonal in the Pauli basis: small code change), σ⁻ decay and a thermal σ± pair (non-diagonal; exact code only at first). Exact L = 12, γ = 0.01, 0.05, 0.2, 1 (≈ 8.5 NH); L ≤ 10 on obelix; trajectories (Hermitian channels) L = 16 on obelix and L = 20 for σˣ and depolarizing at γ = 0.01, 0.05 (≈ 4 NH) | ≈ 13 NH |

Totals:
- Core plan (B1, B2, B3, B4, B5): ≈ 73 NH ≈ 8 200 core-h.
- With B1′: ≈ 120 NH ≈ 13 500 core-h.

For comparison, the last batch (small γ, 2026-10-19) used ≈ 24 100 core-h.

## C. Optional

- **C1. Noise on a single site** (each channel, γ₀ = 1): exact L ≤ 12 on obelix, trajectories L = 16–20. ≈ 6 NH.
- **C2. MPO or MPS single-kick D̄(τ) at L ≈ 100** (new code, reuse julia/lindblad_tdvp.jl): if it reproduces
  single_flip at L = 20–24, it replaces B2 at L = 26 and reaches τ ≈ 20. ≈ 10–30 NH plus development; the γ = 0
  revival must stay exact as a go/no-go test.

## Notes on the new observables

- Trajectory S_A depends on the unravelling (diffusive ≠ jump). Report it as "diffusive-trajectory entanglement".
  S_op, purity, size and support are properties of ρ, so they come from exact runs only (L ≤ 12, 14).
- The size of a trajectory pure state is fixed by the local Bloch vectors, Σ_j (3 − |n_j|²)/4, so no extra
  computation is needed there.
- Check: d ln Tr ρ²/dt = −4γ⟨n_xy⟩, with weights ∝ r_P² (H conserves Σ r_P²).

## Order

1. A1–A4 (laptop).
2. B4, the obelix parts of B3 and B5, and the code changes for B5 and for the observables.
3. B1 and B2 (L = 24).
4. B3.
5. Decide B1′, B2 (L = 26) or C2 from the results.

## Status

- 2026-10-25: plan written. A1–A4 done (`plot_scripts/lean_plan_analysis.py`; outputs `lean_A1_damage.png`,
  `lean_A2_subsampling.csv`, `lean_A3_spread.csv`, `lean_A4_t_e.csv` in `data/used_data/`). Results below.

- 2026-10-25: step B code and scripts ready, nothing submitted. Agreed changes: B1 with 56 diffusive trajectories at
  γ ≤ 0.02 (112 at 0.05), S_A every 1 (γ ≤ 0.01) or 0.5; B3 single_flip with 4 phases × 4 sites; new realizations are
  seeds 2, 3, 4; thermal channel with n̄ = 1. Leonardo: `data/bashes/make_lean_jobs.sh` (15 jobs, ≈ 40 NH expected,
  61 at the caps). obelix: `data/bashes/obelix/` (76 runs, ≈ 15 h, at most ≈ 30 h; start.sh / status.sh / stop.sh /
  pack.sh). Code: `rust/src/noise.rs`, `rust/src/entanglement.rs`, new options of states, states_traj, single_flip
  (CLAUDE.md). Notes: section "Next calculations: the lean data plan" and the new methods in latex/notes.tex.

- Channel study (replaces B4, B5 above): scar, x-Néel and z-Néel under every channel (σᶻ, σˣ, σʸ, depolarizing,
  decay, thermal n̄ = 1), γ = 0.01 … 20 (11 values, γ < 1 the main range). Leonardo: B4 exact L = 12 (all channels,
  operator observables), B5 diffusive L = 20 (Hermitian channels, γ ≤ 2); `make_lean_jobs.sh` now 41 jobs, ≈ 176 NH
  expected, 314 at the caps (B1–B3 unchanged, ≈ 22 NH of it). obelix: `obelix/make_queue.py` → `queue.txt`, 530 runs,
  exact L = 8, 10 (γ = 0, 0.005 … 20), diffusive L = 16, B3 seeds; ≈ 52 h. Zeno-protected runs (σˣ/σʸ on in-plane
  states, σᶻ on z-Néel, at large γ) are cut at a fixed t (CLAUDE.md, "Lean-plan runs").
- Forward/backward (meet-in-the-middle) evolution: `julia/forward_backward.jl`, every channel and the three states,
  validated against exact runs at L = 6 (γ = 0 revivals ≤ 2.5e-6; γ = 0.1 ≤ 9e-6). Not yet run at large L. The
  influence-matrix (transverse) contraction is not implemented (theory only, in the notes).

## Results of A (2026-10-25)

**A1. Lindblad data against the single-kick damage.** F(t) = −(1/γ) ln m⊥ follows ∫₀ᵗ D̄ of the same L, wiggles
included, for every γ ≤ 0.02 at L = 16, 18, 20 (`lean_A1_damage.png`). Diffusive runs (γ = 0.01, 0.02):
F/∫D̄ = 0.90 – 0.98 at t = 4 – 12 (F slightly below, the overlap of kicks); jump runs within ±25 % (noise, amplified
by 1/γ). So the single-kick formula holds for the whole time dependence, not only for Γ_e.
- The slope s cannot be extracted from the Lindblad data at L ≤ 20. Before the damage wraps around the ring
  (t < L/s = 13.8, 15.5, 17.2 for L = 16, 18, 20, with s = 1.16 from D̄ of L = 24 on τ = 4 … 8) the
  period-averaged slope of F covers about one precession period, and D̄ of L ≤ 20 is already bending to its
  saturation there: even the noise-free D̄ of the same L gives s = 0.24 – 0.31 on that window, and the Lindblad
  values scatter between −0.5 and 1 (jump) or 0.17 and 0.71 (diffusive). Also, D̄ becomes L-dependent before L/s
  (τ ≈ 6 for L = 22 against 24), so t < L/s overestimates the L-independent window.
- ∫D̄ against L: ∫₀⁴ D̄ = 27.8, 29.0, 29.7, 30.1, 30.4, 30.6, 30.6 (L = 12 … 24), converged at L ≥ 22;
  ∫₀⁸ D̄ = 64.6, 70.5, 75.8, 79.4, 82.8, 85.0, 86.8, still growing by ≈ 2 % per 2 sites at L = 24.
- Consequence: s and the growth law beyond τ = 8 come only from single_flip at L ≥ 24 (B2) or C2. B1 adds Γ_e at
  L = 22 (the finite-size trend at γ = 0.005 – 0.05), not s.

**A2. single_flip sampling.** The spread is almost all over phases: one run differs from D̄ by up to 16 %
(L = 20) and 17 % (L = 24), the site-averaged D over phases by up to 16 %, the phase-averaged D over sites by only
1.2 % and 0.7 %. Error of D̄ from n_p equally spaced phases × n_s equally spaced sites (worst over τ and over the
offsets of the subsample; L = 20, τ ≤ 25 / L = 24, τ ≤ 8):

| phases × sites | 1 × all | 2 × all | 4 × 1 | 4 × 2 | 4 × 4 | 4 × all | 8 × 4 |
|---|---|---|---|---|---|---|---|
| L = 20 | 26 % | 21 % | 3.4 % | 1.7 % | 1.1 % | 0.5 % | 0.6 % |
| L = 24 | 28 % | 21 % | 1.3 % | 1.1 % | 0.6 % (all 4) | – | < 0.01 % |

- Four equally spaced phases are needed (one or two phases leave ≈ 20 %); with four, four sites give ≤ 1.1 %.
  Eight phases reproduce the 16-phase mean to < 0.01 % (L = 24, 4 sites): the phase dependence has only low harmonics.
- Consequence: B2 with 4 phases × 4 sites = 16 runs, as assumed. B3 single_flip L = 20 can use 4 × 4 instead of
  4 × all sites (≤ 1.1 % instead of 0.5 %): 16 runs instead of 80, ≈ 0.3 NH instead of 1.6 NH per realization.

**A3. Trajectory spread at t_e** (σ₁ = σ_m(t_e) √n; trajectories for a 2 % Γ_e: n = (e σ₁ / 0.02)²):

| runs (L = 16, 18, 20) | σ₁ | n for 2 % Γ_e |
|---|---|---|
| diffusive, γ = 0.01, 0.02 | 0.035 – 0.049 | 23 – 45 |
| diffusive, γ = 0.05 – 0.2 | 0.056 – 0.085 | 58 – 133 |
| diffusive, γ = 0.5 – 10 | 0.09 – 0.16 | 155 – 450 |
| jump, γ = 0.0005 – 0.007 | 0.25 – 0.43 | 1200 – 3500 |

- σ₁ does not depend on L between 16 and 20.
- The diffusive rule is stronger than assumed: the variance ratio jump/diffusive at t_e is ≈ 60 – 100 (not
  15 – 20), so jump trajectories cost ≈ 30× more per error bar even after the first-kick saving.
- Consequence: 112 diffusive trajectories give ≈ 1 % at γ ≤ 0.02 and ≈ 1.5 – 2 % at γ = 0.05; 56 would still give
  ≤ 2 % at γ ≤ 0.02 (the bench showed 28 trajectories on 112 cores at 0.94 efficiency at L = 20).

**A4. t_e of the planned runs** (upper bound: 1/Γ_e(L = 20); single kick with D̄ of the same L, continued flat or
linearly with s past τ = 8):

| run | γ | t_e from L = 20 | t_e single kick | t-max = 1.5 t_e |
|---|---|---|---|---|
| B1, L = 22 | 0.005 | 15.9 | 14.0 – 15.4 | 23.9 |
| B1, L = 22 | 0.01 | 9.4 | 8.9 – 9.0 | 14.1 |
| B1, L = 22 | 0.02 | 5.8 | 5.6 | 8.7 |
| B1, L = 22 | 0.05 | 3.2 | 3.0 | 4.7 |
| B1′, L = 24 | 0.01 | 9.4 | 8.8 | 14.1 |
| B1′, L = 24 | 0.02 | 5.8 | 5.5 | 8.7 |

- The t-max of the plan (24, 14, 9, 5) hold. At γ = 0.005 the single-kick t_e needs D̄ past τ = 8 (extrapolated).
- Cost options for B1 (≈ 1500 s per unit of t with 112 trajectories, 52 units of t in all): as planned ≈ 22 NH;
  with 56 trajectories at γ ≤ 0.02 (≤ 2 %) ≈ 12 NH.
