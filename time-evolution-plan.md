# Time evolution: scar vs thermal decay under dephasing

The previous plan (probes P1/P2/P3 and Liouvillian ED, file `scar-vs-thermal-decay-plan.md`) is in git history.

## Goal

Compare the decay of local magnetization under uniform σᶻ dephasing for the scar state and for a thermal
(non-scar) reference, and extract the thermodynamic rate Γ_∞(γ) = lim_{1/L → 0} Γ(L, γ) for both.
Question: does the scar lifetime diverge as γ → 0 (and with which power of γ), while the thermal rate stays
finite?

## Setup

- Exact evolution of ρ with the `states` binary: full Pauli basis, random c (one realization per L,
  `Params::random_c`), coefficients from `rust/src/parameters.rs` (h = 1, a = b = (1, 1, 1), c = 𝟙 × random factors).
- Lindblad equation and conventions as in CLAUDE.md: dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ).
- γ grid: 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10.
- Sizes: L = 4 … 12 done (Néel states even L only); L = 14 on Leonardo.

## States and signals

| State | `--state` | ρ₀ | Signal | At H_I = 0 |
|---|---|---|---|---|
| Scar | `minus-x` | \|−⟩^⊗L | m_⊥ = √(m_x² + m_y²) | e^(−2γt) |
| Thermal reference | `neel-x` | \|+−+−…⟩ (even L) | s_⊥ = √(s_x² + s_y²) | e^(−2γt) |
| z-Néel (done, biased) | `neel` | \|↑↓↑↓…⟩ (even L) | \|s_z\| | no decay |

m_α = (1/L) Σ_j ⟨σ^α_j⟩, s_α = (1/L) Σ_j (−1)^j ⟨σ^α_j⟩.

- Why x-Néel and not z-Néel: z-Néel is a fixed point of the dissipator (diagonal in the z basis, D(ρ₀) = 0) and
  s_z is invisible to it (D†(σᶻ_j) = 0). The bath acts only after H_I rotates weight into σˣ/σʸ, so with H_I = 0
  s_z never decays and at large γ it is Zeno-protected. The x-Néel s_⊥ has the same e^(−2γt) baseline as the
  scar's m_⊥.
- x-Néel has zero tower overlap of its observable: P_tower O_π P_tower = 0 for O_π = Σ_j (−1)^j σ^α_j.
  It is not a dark state of 𝓛 (only |↑…↑⟩ and |↓…↓⟩ are).

## Rate extraction (`plot_scripts/fit_states.py`, `fit_states_inverse_L.py`)

- Envelope E(t) = max_{t' ≥ t} signal(t') (non-increasing; removes precession zeros and revivals).
- Γ_e = 1/t_e, with E(t_e) = 1/e. Main rate.
- Tail: slope of ln E between 1e-2 and 1e-3 (Γ_tail). Picks up the slow modes (tower edges for the scar).
- Runs stop once the signal stays below 1e-3 (`--stop`); γ = 0 runs to `--t-zero`.
- Thermodynamic limit: Γ_e(L) = A + B/L fitted on L ≥ 8, Γ_∞ = A. Check: quadratic in 1/L on L ≥ 6;
  error = max(fit error, |linear − quadratic|). Néel at γ ≤ 0.02: drop L = 6, 8 (finite-size revivals).

## Next steps

1. Done (2026-10-16 batch, now in `data/used_data/`): x-Néel L = 4 … 14 exact; scar L = 14 exact (γ ≤ 0.02 cut at 24 h);
   trajectories L = 12 … 20 for scar and x-Néel; measured costs in CLAUDE.md (Cluster).
2. Done: Γ_∞(γ) scar vs x-Néel (`states_thermodynamic_rate.png`, with γ = 0); results in CLAUDE.md ("Results up
   to L = 20"). L = 22 trajectories not run.
3. Done (batch finished 2026-10-20; all data in `data/used_data/`): small-γ
   scar runs and the single-kick test (below). Γ_e(L, γ) follows exp[−γ ∫ D̄] at finite L (measured/predicted Γ_e mostly 0.95 – 1.05), and the
   cap Γ_e/γ → L − 1 is D_sat ≈ L − 1; the collapse gives ν = 0.24 – 0.48 and D̄ of L = 22, 24 grows roughly
   linearly up to τ = 8, so ballistic vs diffusive spreading is still open. Results in CLAUDE.md ("Results at small
   γ and the single-kick test"). Next: D̄(τ) at L ≥ 24 for τ > 8 (`single_flip` with more sites, or MPS).

## Small-γ scar: single-kick test

Question: at small γ, is the decay of m_⊥ set by isolated σᶻ kicks, each damaging the coherence as one flip
evolved with H does?

Prediction. One kick on site k at time t′ (phase φ = ht′) gives, after a time τ under H, the damage
D(τ) = Σ_j [1 − Re c_j(τ)], c_j = (⟨σˣ_j⟩ + i⟨σʸ_j⟩) / (reference-1 value at t′ + τ) (`single_flip`). Averaged
over the kicked site and the phase, D̄(τ) gives

  m_⊥(t) ≈ exp[−γ ∫₀ᵗ D̄(τ) dτ].

D̄(0) = 2, so the initial rate is the free-spin 2γ; D̄ grows as the flip spreads and saturates at D_sat(L). If
D_sat ≈ L − 1, this reproduces the finite-size cap Γ_e/γ → L − 1 seen in the γ ≤ 0.02 data (CLAUDE.md, "Results up
to L = 20"); the L-independent part of the question is whether D̄(τ) grows linearly (ballistic spreading) before it
saturates, which `single_flip` at L = 22, 24 checks.

Runs (generators and costs: CLAUDE.md, Cluster, "Small-γ scar runs"):
- (A) `single_flip`: L = 12 … 20, 16 phases × all sites, τ ≤ 25 (until D saturates); L = 22, 24, 16 phases ×
  sites 0, L/4, L/2, 3L/4, τ ≤ 8.
- (B) jump trajectories (`states_traj --unravel jump`, output step 0.25): L = 12 … 20, γ = 0.0005 … 0.007
  (√2 spacing; ×2 below 0.002 at L = 18, 20), t-max = 3/(γ(L − 1)), 448 trajectories (224 at L = 18, 20,
  γ < 0.002), split over `--traj-seed` where needed. Same random-c realization (`--seed 1`) as (A).

References needed for the comparison:
- Analytic: reference 1 (γ = 0: m_⊥ = 1, c_j = 1 without a kick); reference 2 (H_I = 0: m_⊥ = e^(−2γt), D = 2 at
  all τ); D(0) = 2. All three are unit tests (`rust/tests/checks.rs`).
- Exact ρ at small γ (new, `make_small_gamma_exact_jobs.sh`): L = 8, 10 on the whole γ grid, L = 12 for
  γ ≥ 0.002, same t-max: checks the jump trajectories at small γ, where they have not been validated yet.
- Diffusive cross-check (new, in tier 1): L = 12, 14, γ = 0.002, 0.005, output step 0.1: jump vs diffusive at the
  production sizes.
- Existing γ = 0.01 data (`data/used_data/`): exact L ≤ 14 and diffusive trajectories L = 12 … 20 (not rerun; γ = 0.01
  is left out of the new grids). Anchors the small-γ points to the earlier rates.
- Done: jump vs exact at L = 8, γ = 0.1 and 0.01 (2000 trajectories, rms deviation ≈ 1 standard error).

Expected: Γ_e/γ between 2 (initial rate) and ≈ L − 1 (saturated damage), approaching L − 1 as γ → 0 at fixed L;
at L = 8 a laptop `single_flip` run gives D̄ ≈ 6.7 – 8.2 for τ ≥ 5.

Analysis (`plot_scripts/`, default `data/used_data`): `states_data.py` pools the traj-seed files of one (L, γ)
weighted by their trajectory counts; `fit_states.py` then `fit_small_gamma.py` (Γ_e/γ against L with statistical
errors, scaling collapse, jump vs exact/diffusive cross-checks); `fit_states_inverse_L.py` then `single_kick.py` (D̄(τ) from the `single_flip` CSV,
mean over m and k; m_⊥(t) against exp[−γ ∫ D̄] at the same L, Γ_e against the prediction, Γ_∞ against the
prediction from D̄ of L = 24).
