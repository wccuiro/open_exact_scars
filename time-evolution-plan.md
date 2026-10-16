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

1. x-Néel runs, L = 4 … 14 (even), full γ grid, on Leonardo; record time and memory per run (L = 14 is new:
   matrix-free H application, dimension 4^14).
2. Scar at L = 14 if the benchmark allows.
2b. Trajectories (`states_traj`, `make_jobs.sh`), scar and x-Néel, all γ: L = 12, 14 (check against exact),
   16, 18, 20; L = 22 after the L = 20 timings. Γ_e(L) from these extends the 1/L extrapolation.
3. Γ_∞(γ) scar vs x-Néel on the same figure (`states_thermodynamic_rate.png`); compare the small-γ behaviour
   (scar: Γ_∞ ∝ γ^(≈1/2) for 0.05 ≤ γ ≤ 0.2) with the thermal reference.
