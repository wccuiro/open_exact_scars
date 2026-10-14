# Scar vs thermal decay under dephasing: measurement plan

## Goal

Decide whether the L-dependence of the dephasing-induced decay rate of the scar is specific to the scar or generic for this Hamiltonian. Existing data (L = 4–8, γ = 0.01) show a scar decay rate roughly γ(L−1).

Method: compare the scar against two non-scar probes evolved with the same Lindbladian, then confirm with the Liouvillian spectrum.

## Model

- H = H_I + h S^z_tot
- H_I = Σ_j [ Σ_α ( a_α R_{j,j+1} σ^α_{j+2} + b_α σ^α_{j−1} R_{j,j+1} ) + Σ_{α,β} c_{αβ} σ^β_{j−1} R_{j,j+1} σ^α_{j+2} ]
- R_{j,j+1} = (𝟙 − σ_j·σ_{j+1})/4 is the bond singlet projector.
- Periodic boundary conditions.
- Couplings: a_α and b_α are uniform. Only c_{αβ} is random per site (the realization files store c_xz, c_yx, c_zy and the initial-state angles φ_j). Use the same stored realization for every probe at a given L.
- Lindbladian: 𝓛ρ = −i[H, ρ] + γ Σ_j (σ^z_j ρ σ^z_j − ρ). With this convention a single-spin coherence decays at 2γ.
- Steady state: 𝟙/2^L.

### Exact identities (use as code checks)

1. For a Pauli string P, D†(P) = −2γ n_xy(P) P, where n_xy is the number of sites carrying σ^x or σ^y.
2. D†(S^±) = −2γ S^±. With H_I = 0 (set a = b = c = 0), m_⊥(t) = e^{−2γt} exactly for every L. Run this check first.
3. [σ^z_j, S^z_tot] = 0, so the lift term is invisible to the dissipator.
4. S^z_tot is not conserved by H_I. In the γ = 0 data, mz relaxes only partially: at L = 5, θ = π/2 it goes from 0.279 to a mean of about 0.15 for t > 20. The residual should be the initial state's tower component, which keeps its S^z. Check this against Tr(P_tower S^z ρ₀).
4b. Conserved quantities of the closed system: H, P_tower, and every tower projector |Φ_n⟩⟨Φ_n| (the commutant). Every probe must either live entirely in the tower (P1) or have the tower part removed (P2, P3).
5. Tower weight: for ρ = |−⟩⟨−|^⊗L, dw/dt at t = 0 equals −γ(L−1), where w = Tr(P_tower ρ) and P_tower projects onto S = L/2.

## Probes

| Probe | Initial object | Observable | At γ = 0 | Question at γ > 0 |
|---|---|---|---|---|
| P1 Scar | ρ₀ = \|−⟩⟨−\|^⊗L | m_⊥ = 2\|⟨S^+⟩\|/L | stays 1, oscillates | Γ_scar(L, γ) |
| P2 Energy | H̃ = H − h S^z P_tower (Heisenberg picture) | C_H̃(t) | ≡ 1, conserved | Γ_H̃(L, γ) |
| P3 Thermal | O_π (Heisenberg picture) | C_π(t) | decays at Γ₀(L) | ΔΓ = Γ − Γ₀ |

### P1: Scar reference

- Observable: m_⊥(t) = 2|⟨S^+⟩|/L. This removes the h-precession, so no fitting through oscillations is needed.
- Also record the tower weight w(t) = Tr(P_tower ρ(t)), and check its initial slope against −γ(L−1).
- Existing data (data/disorder_L*.csv) cover only γ = 0.01. Use them as a validation point; the γ grid needs new runs. Existing fits (window t ∈ [5, 60]): Γ/γ = 3.22, 4.01, 4.69, 5.67, 6.51 for L = 4…8.

### P2: Energy (non-scar, conserved when closed)

- Remove the tower part of H. P_tower commutes with H and P_tower H P_tower = h S^z P_tower, since H_I annihilates the tower. So H̃ = H − h S^z P_tower is still exactly conserved at γ = 0 and has no tower component. Using H directly would mix in a tower piece of weight h² L(L+1)(L+2)/12 relative to Tr H², which decays at the scar rate and contaminates the fit.
- Compute C_H̃(t) = Tr[H̃ e^{𝓛†t}(H̃)] / Tr[H̃²]. This is the linear-response limit of ρ₀ ∝ 𝟙 − βH̃. H̃ is Hermitian, so evolving H̃/2^L forward with 𝓛 gives the same autocorrelation.
- Prediction for the initial rate: Γ_H̃(0) = (γ/2) Σ_j ‖[σ^z_j, H̃]‖²_F / ‖H̃‖²_F. This is O(γ) and L-independent. For H itself it reduces to 2γ Σ_P n_xy(P)|h_P|² / Σ_P |h_P|², where H = Σ_P h_P P. The h S^z term has n_xy = 0, so it lowers the rate. Report h.
- Compare the full decay against this. It may be non-exponential, because weight moved into S^z-like content is protected by the dissipator until H_I rotates it out.

### P3: Thermal reference (non-scar, thermalizes when closed)

- Main operator: O_π = Σ_j (−1)^j σ^x_j, for even L only (L = 4, 6, 8; L = 10 optional if affordable). It is Hermitian, so one real run per (L, γ).
- Why k = π and not k = 2π/L:
  - k = 2π/L changes with L, so an L-dependence of Γ_k would mix with k-dependence.
  - Because a_x, b_x are uniform, σ^x_k overlaps the momentum-k energy density H_k = Σ_j e^{ikj} h_j. Energy is conserved, so that component relaxes diffusively at rate ∝ D k². At k = 2π/L this gives a slow, L-dependent hydrodynamic contribution, which is exactly the contamination to avoid. At k = π the diffusive mode is not slow.
- Optional secondary probe: k = 2π/L, only as a diagnostic of that diffusive contribution. Odd L with k closest to π is not needed.
- Zero tower overlap is exact: P_tower O_k P_tower = 0 for k ≠ 0, because P_tower σ^x_j P_tower = (2/L) S^x P_tower for every j. This also gives zero overlap with every conserved |Φ_n⟩⟨Φ_n|. The k = 0 mode is S^x, the scar observable.
- Overlap with H: Tr(O_k† H) = 0 exactly for k ≠ 0. The c terms contain no single-site σ^x, and the single-σ^x content of H, (a_x + b_x)/4 · Σ_j σ^x_j, is uniform. Keep the projection Õ_k = O_k − H Tr(H O_k)/Tr(H²) only as a guard, and record that it is zero.
- Compute C_π(t) = Tr[O_π e^{𝓛†t}(O_π)] / Tr[O_π²].
- At γ = 0, fit the early exponential decay Γ₀(L) before the finite-size noise floor.
- At γ > 0, extract Γ(γ, L) and the slope dΓ/dγ. The naive value is 2γ, since n_xy = 1 for a bare σ^x. Operator spreading under H_I raises the effective n_xy and may make the slope depend on L (see the literature on operator growth in open systems, e.g. Schuster & Yao, PRL 2023). Measuring this is the point of the probe.

## Liouvillian spectrum

- Full exact diagonalization only (no Krylov or Arnoldi methods). Build the dense 4^L × 4^L Liouvillian with the existing Rust ED code and compute all eigenvalues and right eigenvectors.
- Feasible sizes (complex double, matrix only; eigenvectors need about the same again):
  - L = 4, 5, 6: 256, 1024, 4096. Trivial.
  - L = 7: 16384, about 4.3 GB per matrix. Feasible on a workstation (≈ 10 GB total, runtime of hours).
  - L = 8: 65536, about 69 GB per matrix, ≈ 140 GB with eigenvectors. Only if such a machine is available; otherwise the spectrum stops at L = 7 and L = 8 is covered by the dynamics.
- Disorder in c removes translation symmetry, and H_I breaks S^z and spin-flip parity, so there is no block decomposition to exploit.
- For each right eigenvector r_n (reshaped to a matrix), compute:
  - tower weight w_n = ‖P r_n P‖_F / ‖r_n‖_F
  - mixed weight ‖P r_n (1−P)‖_F / ‖r_n‖_F
  - energy overlap |Tr(H r_n)| for the left eigenvector
- Classify modes as tower (w ≈ 1), mixed, or bulk (w ≈ 0).
- Expectation: tower coherences |S,m⟩⟨S,m'| show Im λ ≈ h(m − m').
- Plot −Re λ / γ against L for each class and each γ, with the slowest bulk mode highlighted.
- Order of limits (the project question): at fixed L, the bulk gap should vanish as γ → 0. For L → ∞ first, a finite bulk gap (Ruelle–Pollicott) is expected. Track the bulk gap Δ(γ, L) on the γ grid to see the crossover.

## Parameters

- P1, P2: L = 4, 5, 6, 7, 8. P3: L = 4, 6, 8 (10 optional).
- γ ∈ {0, 0.1}
- h as in the current runs.
- t_max ≈ 5/Γ for each run, with a dt that resolves the precession (period 2π/h).
- One fixed disorder realization per L for every probe. If time allows, 3–5 realizations for L ≤ 7 to get error bars; do this last.

## Run order

1. Sanity check: P1 with H_I = 0 must give exactly e^{−2γt}.
2. P1 over the γ grid (check against the existing γ = 0.01 data).
3. P2 (H̃).
4. P3 (k = π).
5. Spectrum.
6. Extra disorder realizations.

## Fitting

- Fit log of the observable linearly on a window starting after the initial transient (t ≳ 5) and ending when the signal reaches about 1e-3 or the noise floor. Choose the window automatically and record it.
- Save Γ, its fit uncertainty, and the window for every (probe, L, γ, k).

## Outputs

- One CSV per probe with columns L, γ, (k), t, observable.
- Summary table rates.csv with columns probe, L, γ, k, Γ, σ_Γ, t_fit_min, t_fit_max.
- Main figure: Γ/γ against L for P1, P2 and P3 (ΔΓ/γ) on shared axes, one panel per γ.
- Spectrum figure: −Re λ/γ against L, colored by tower weight.
- Sanity-check figure: P1 with H_I = 0 against e^{−2γt}.

## How to read the result

- If Γ_scar/γ grows ∝ L while Γ_H/γ and dΓ_k/dγ stay flat in L, the L-scaling is specific to the scar.
- If the thermal slope also grows with L, the scaling is a generic operator-size effect of dephasing in this model.
- The spectrum should show the same split between tower-class and bulk-class eigenvalues.