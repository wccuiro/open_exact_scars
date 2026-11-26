# Floquet circuit with a uniform field layer

One period on a periodic ring of `N ≥ 4` sites applies the shifted interaction layers first, one uniform field rotation second, and independent on-site dissipation last. Number the sites **1 through N** and wrap all site indices around the ring.

```text
Order of application:  U_d → … → U₂ → U₁ → field on every site → noise on every site
```

## Construct the interaction layers

1. Choose `d` as the smallest divisor of N that is at least four. Use four layers when N is divisible by four. Otherwise use the smallest divisor greater than four. The spacing keeps the four-site blocks disjoint, and the divisor makes the pattern close around the ring.
2. In layer `Uℓ`, start blocks at `j = ℓ, ℓ+d, …, ℓ+(N/d−1)d`. Each block covers `(j, j+1, j+2, j+3)`. The following `d−4` sites are idle in that layer. Shift the block positions by one site from U₁ to U₂ and so on. Bond-dependent coefficients stay attached to their physical bonds.
3. Build each block from the RA, AR and ARA terms associated with its middle bond. In the table, the two R entries are the two ends of **one** singlet projector. A denotes a single-site Pauli factor and I the identity.

   | Term | Site j | Site j+1 | Site j+2 | Site j+3 |
   |---|:---:|:---:|:---:|:---:|
   | RA | I | R | R | A |
   | AR | A | R | R | I |
   | ARA | A | R | R | A |

4. Sum over spin axes using the model coefficients `a`, `b` and `c` for RA, AR and ARA respectively. The block gate is **one exponential**, `Vⱼ = exp[−i Δt (RA + AR + ARA)ⱼ]`. Multiply the `N/d` disjoint gates to form Uℓ. Each middle bond appears once per period at its original strength. [Bond Hamiltonian](rust/src/model.rs)

## Apply the field and dissipation

Immediately after U₁, apply the same field rotation on every site in parallel. With `Bⱼ = (h/2) Zⱼ`, the product of all N rotations is the collective rotation `U_B = exp[−i h Δt Sᶻ_tot]`. Every wire receives the full angle `h Δt` once, including wires that were idle in U₁.

If dissipation is present, apply one final parallel layer. On wire j use the channel `Eⱼ = exp(Δt Dⱼ)`, where Dⱼ includes the local noise rate. Independent on-site generators commute across sites, so their combined channel is exactly `E = E₁ ⊗ ⋯ ⊗ E_N`. For Z dephasing, each Eⱼ multiplies the local X and Y components by `exp(−2γΔt)` and leaves I and Z fixed. [Noise rates](rust/src/noise.rs)

In multiplication order, the unitary is `U = U_B U₁ U₂ ⋯ U_d`. One time step sends `ρ → U ρ U†`, followed by E when noise is present. The circuit has **d + 1** parallel layers for unitary evolution and **d + 2** with the separate noise row.

### Why the uniform field preserves the scar tower

The middle-bond projector `R = (I − XX − YY − ZZ)/4` annihilates every state in the symmetric scar tower. The outer Pauli factors commute with R because they act on other sites. Hence every interaction block Hamiltonian annihilates the tower and its exponential acts as the identity there. The collective field rotation preserves the tower and supplies its exact phases. At zero dissipation the circuit therefore gives the exact tower dynamics for any Δt. Away from the tower the product is a first-order Trotter approximation. On-site noise can transfer weight out of the tower.

## Worked examples

Each entry under **starts** lists the first site of each four-site block. Read the interaction rows from bottom to top, skipping absent layers, then apply the field and optional noise on every site.

| Layer | N = 8 starts | N = 10 starts | N = 10 idle sites | N = 12 starts | N = 14 starts |
|---|---|---|---|---|---|
| U₁ | 1, 5 | 1, 6 | 5, 10 | 1, 5, 9 | 1, 8 |
| U₂ | 2, 6 | 2, 7 | 1, 6 | 2, 6, 10 | 2, 9 |
| U₃ | 3, 7 | 3, 8 | 2, 7 | 3, 7, 11 | 3, 10 |
| U₄ | 4, 8 | 4, 9 | 3, 8 | 4, 8, 12 | 4, 11 |
| U₅ | — | 5, 10 | 4, 9 | — | 5, 12 |
| U₆ | — | — | — | — | 6, 13 |
| U₇ | — | — | — | — | 7, 14 |

On **eight sites**, the first layer applied is U₄, with blocks on `(4,5,6,7)` and `(8,1,2,3)`. On **ten sites**, the first is U₅, with blocks on `(5,6,7,8)` and `(10,1,2,3)`. In the latter block RA covers `(1,2,3)`, AR covers `(10,1,2)`, and ARA covers all four. Sites 4 and 9 are idle during U₅. Both receive their B rotation in the field layer.

## Draw the circuit

1. Keep sites 1 through N in fixed vertical columns. Draw time upward, with U_d at the bottom and U₁ at the top of the interaction stack. Bracket each interaction layer on the left and label it Uℓ.
2. Inside each brace draw **three rows**, RA, AR and ARA from top to bottom. For every block, draw a rounded capsule across exactly the wires occupied by each term in the placement table. The three rows represent summands of the same block Hamiltonian.
3. Use one color per operator family throughout the figure. Label RA in teal, AR in gold, ARA in purple and B in blue. Mark each wire outside all blocks in a layer with a dashed vertical outline labeled **idle** across the three rows.
4. Split any capsule crossing the periodic boundary from site N to site 1 at the two diagram edges. Use the same label and color on both pieces, with dashed ends at the cut. Count the pair as one term.
5. Draw **one Field row immediately above U₁**, outside every brace. Place a blue circle labeled **B** on every wire at the same height. For a dissipative period, extend all wires one row higher. Label that row **Noise** and place a single-site box labeled **E** on each wire. This top row acts last.
6. Check that every interaction brace contains `N/d` capsules of each type (RA, AR and ARA) and that each block start occurs exactly once across the period. Count exactly N circles in the Field row and, when present, N boxes in the Noise row.

## Implementation status

The Rust evolution follows this guide (since 2026-11-25). For each site, the field rotation and the noise channel are folded into the **last** interaction gate that touches it, applied after that gate's interaction. Neither commutes with that gate, but both commute with every gate applied later, so the product of the gates equals `E ∘ U_B U₁ ⋯ U_d` exactly. At γ = 0 the scar state reproduces reference 1 to round-off for any Δt (tested at L = 6, 8, 10, 14). The TikZ generator draws the circuits with the Field and Noise rows above U₁. The notes chapter "Floquet (Trotterized) dynamics" (`latex/sections/floquet.tex`) shows the circuits for N = 8, 10, 12 and 14, with checks against the exact evolution. [Rust evolution](rust/src/floquet.rs) · [TikZ generator](plot_scripts/floquet_circuit_tikz.py)

The previous protocol (commit f223cc3) used the smallest disjoint spacing rather than this shift pattern, for example five layers at N = 14. It applied field and noise first. Its runs of 2026-11-24 are in `data/non_used_data/floquet_old_protocol/`.
