# Lindblad evolution of ρ(0) = |ψ₀⟩⟨ψ₀|, |ψ₀⟩ = |−⟩^⊗L, with ITensor: ρ as an MPS, evolved by TDVP.
#
#     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ)        (model and conventions: ../CLAUDE.md)
#
# Representation: ρ = 2^(−L) Σ_P r_P P over Pauli strings, so every site carries the 4 states
# (1, σˣ, σʸ, σᶻ) and the MPS holds the real coefficients r_P. Left multiplication by σ^a acts on
# them as the matrix A^a + i B^a (A^a, B^a real), right multiplication as A^a − i B^a, hence
#
#     −i[P, ·] = 2 Im ⊗_j (A^{a_j} + i B^{a_j})     for a Pauli string P = ⊗_j σ^{a_j},
#
# a real operator; dephasing is diagonal, −2γ on σˣ and σʸ. The Lindbladian is therefore a real
# MPO (periodic boundaries included as long-range terms) and the whole evolution is real.
#
# Usage:
#     julia --project=. lindblad_tdvp.jl L maxdims [gamma=0.01] [tmax=100] [dt=0.1] [cutoff=1e-10]
#     julia --project=. lindblad_tdvp.jl 8 8,16,32
#
# Output, one file per bond dimension: ../data/tdvp_L<L>_chi<maxdim>_dt<dt>.csv with columns
#     L, gamma, maxdim, t, mx, my, mz, bond_dim, seconds
# m_α = (1/L) Σ_j ⟨σ^α_j⟩, bond_dim = largest bond dimension in use, seconds = wall time so far.

using ITensors, ITensorMPS, LinearAlgebra, Printf

# ---- the 4-dimensional site: coefficients of (1, σˣ, σʸ, σᶻ) ---------------------------------

ITensors.space(::SiteType"PauliVec") = 4
ITensors.state(::StateName"I", ::SiteType"PauliVec") = [1.0, 0.0, 0.0, 0.0]
ITensors.state(::StateName"X", ::SiteType"PauliVec") = [0.0, 1.0, 0.0, 0.0]
ITensors.state(::StateName"Y", ::SiteType"PauliVec") = [0.0, 0.0, 1.0, 0.0]
ITensors.state(::StateName"Z", ::SiteType"PauliVec") = [0.0, 0.0, 0.0, 1.0]
# |−⟩⟨−| = (1 − σˣ)/2, normalized as a vector
ITensors.state(::StateName"Minus", ::SiteType"PauliVec") = [1.0, -1.0, 0.0, 0.0] / sqrt(2)

# σ^a σ^b = Σ_c (A^a + i B^a)[c, b] σ^c   (rows: output, columns: input)
ITensors.op(::OpName"Ax", ::SiteType"PauliVec") = [0.0 1 0 0; 1 0 0 0; 0 0 0 0; 0 0 0 0]
ITensors.op(::OpName"Ay", ::SiteType"PauliVec") = [0.0 0 1 0; 0 0 0 0; 1 0 0 0; 0 0 0 0]
ITensors.op(::OpName"Az", ::SiteType"PauliVec") = [0.0 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 0]
ITensors.op(::OpName"Bx", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 0 0; 0 0 0 -1; 0 0 1 0]
ITensors.op(::OpName"By", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 0 1; 0 0 0 0; 0 -1 0 0]
ITensors.op(::OpName"Bz", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 0]
# number of σˣ, σʸ on the site: dephasing is −2γ times this
ITensors.op(::OpName"Deph", ::SiteType"PauliVec") = [0.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]

const AXIS = ("x", "y", "z")

# ---- model -------------------------------------------------------------------------------------

"""
H = H_I + h Sᶻ_tot as a sum of Pauli strings (periodic chain, baseline coefficients of CLAUDE.md).
Returns a Dict: sorted list of (site => axis), axis 1, 2, 3 = x, y, z  →  real coefficient.
"""
function hamiltonian_terms(L; h = 0.5, a = (0.73, 0.41, 0.29), b = (0.58, -0.37, 0.66))
    L >= 4 || error("periodic boundaries need L >= 4")
    # c[α, β] multiplies σ^β_{j−1} R_{j,j+1} σ^α_{j+2}
    c = zeros(3, 3)
    c[1, 3], c[3, 2], c[2, 1] = 0.31, -0.44, 0.27

    terms = Dict{Vector{Pair{Int, Int}}, Float64}()
    add!(coef, string) = (key = sort(string); terms[key] = get(terms, key, 0.0) + coef)
    for j in 1:L
        left, s1, s2, right = mod1(j - 1, L), j, mod1(j + 1, L), mod1(j + 2, L)
        # coef · R_{s1,s2} · dressing, with R = (1 − σ_{s1}·σ_{s2}) / 4
        function dressed!(coef, dressing)
            coef == 0 && return
            add!(coef / 4, dressing)
            for δ in 1:3
                add!(-coef / 4, [s1 => δ, s2 => δ, dressing...])
            end
        end
        for α in 1:3
            dressed!(a[α], [right => α])
            dressed!(b[α], [left => α])
            for β in 1:3
                dressed!(c[α, β], [left => β, right => α])
            end
        end
        add!(h / 2, [j => 3])
    end
    return filter(t -> t.second != 0, terms)
end

"The Lindbladian as an OpSum over the PauliVec sites."
function lindbladian(L, γ)
    os = OpSum()
    for (string, coef) in hamiltonian_terms(L)
        k = length(string)
        # 2 Im Π_j (A_j + i B_j): every way of taking B on an odd number of sites
        for mask in 1:(2^k - 1)
            nB = count_ones(mask)
            isodd(nB) || continue
            ops = Any[]
            for (i, (site, axis)) in enumerate(string)
                push!(ops, (((mask >> (i - 1)) & 1 == 1) ? "B" : "A") * AXIS[axis], site)
            end
            os += (2 * coef * (-1)^((nB - 1) ÷ 2), ops...)
        end
    end
    for j in 1:L
        os += (-2γ, "Deph", j)
    end
    return os
end

# ---- measurement -------------------------------------------------------------------------------

"(m_x, m_y, m_z) with m_α = (1/L) Σ_j Tr[σ^α_j ρ] / Tr ρ."
function magnetization(psi::MPS)
    L = length(psi)
    s = siteinds(psi)
    # Tr[P ρ] is the coefficient r_P: contract with 1 on every site except the measured one
    left = Vector{ITensor}(undef, L + 1)
    right = Vector{ITensor}(undef, L + 1)
    left[1] = ITensor(1.0)
    right[L + 1] = ITensor(1.0)
    for j in 1:L
        left[j + 1] = left[j] * psi[j] * state(s[j], "I")
    end
    for j in L:-1:1
        right[j] = right[j + 1] * psi[j] * state(s[j], "I")
    end
    trace = scalar(left[L + 1])
    m = zeros(3)
    for j in 1:L, (α, name) in enumerate(("X", "Y", "Z"))
        m[α] += scalar(left[j] * psi[j] * state(s[j], name) * right[j + 1])
    end
    return m ./ (L * trace)
end

# ---- evolution ---------------------------------------------------------------------------------

function evolve(L, γ, maxdim; tmax = 100.0, dt = 0.1, cutoff = 1.0e-10, datadir = joinpath(@__DIR__, "..", "data"))
    sites = siteinds("PauliVec", L)
    liouvillian = MPO(lindbladian(L, γ), sites)
    psi = MPS(sites, "Minus")
    nsteps = round(Int, tmax / dt)

    mkpath(datadir)
    path = joinpath(datadir, "tdvp_L$(L)_chi$(maxdim)_dt$(dt).csv")
    start = time()
    open(path, "w") do io
        println(io, "L,gamma,maxdim,t,mx,my,mz,bond_dim,seconds")
        for step in 0:nsteps
            if step > 0
                psi = tdvp(
                    liouvillian, dt, psi;
                    nsteps = 1, nsite = 2, maxdim, cutoff, normalize = false, outputlevel = 0,
                    updater_kwargs = (; ishermitian = false, issymmetric = false, tol = 1.0e-10, krylovdim = 30, eager = true),
                )
            end
            mx, my, mz = magnetization(psi)
            @printf(io, "%d,%g,%d,%.6f,%.12e,%.12e,%.12e,%d,%.2f\n", L, γ, maxdim, step * dt, mx, my, mz,
                maxlinkdim(psi), time() - start)
        end
    end
    @printf("L = %d, gamma = %g, maxdim = %d: Lindbladian MPO bond dimension %d, final MPS bond dimension %d, %.1f s for %d steps\n",
        L, γ, maxdim, maxlinkdim(liouvillian), maxlinkdim(psi), time() - start, nsteps)
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: julia --project=. lindblad_tdvp.jl L maxdims [gamma] [tmax] [dt] [cutoff]")
    L = parse(Int, ARGS[1])
    maxdims = parse.(Int, split(ARGS[2], ","))
    γ = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.01
    tmax = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 100.0
    dt = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.1
    cutoff = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1.0e-10
    for maxdim in maxdims
        evolve(L, γ, maxdim; tmax, dt, cutoff)
    end
end
