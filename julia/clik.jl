# CLIK (Westhoff, Moroder, Schollwöck & Paeckel, Phys. Rev. X 16, 041020 (2026)): eigenvalues of the
# Lindbladian from time-evolved states, without diagonalizing 𝓛. Implementation guide (written for another
# repository, followed here): ../data/docs/CLIK_MPS/IMPLEMENTATION.md; "Eq.", "Alg." refer to the paper.
#
#     dρ/dt = 𝓛ρ = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ)        (model and conventions: ../CLAUDE.md)
#
# Two evolutions, sampled every δt up to T and stored in lockstep (p₀, u₀, p₁, u₁, …), each state normalized:
#   physical   ρ_p, trace 1, real time z = t                 → steady state and slow modes
#   traceless  ρ_u, trace 0, complex time z = t e^{−iα}      → slowest excited modes; mode k is damped at
#              ξ_k = −Re λ_k cos α − Im λ_k sin α (Eq. 7), so Im λ > 0 gains weight. Needs α < α_max.
# 𝓛 is projected on their span: M_ij = ⟪ρ_i|ρ_j⟫, L_ij = ⟪ρ_i|𝓛|ρ_j⟫, and after deflating the Gram matrix
# (eigenvalues below ε dropped) the projected problem is diagonalized (Alg. 2). The Krylov space of a shorter T
# is a leading block, so one run gives the eigenvalues for a list of nested T.
#
# Representation as in lindblad_tdvp.jl: ρ = 2^(−L) Σ_P r_P P, PauliVec sites (1, σˣ, σʸ, σᶻ), 𝓛 a real MPO.
# ⟪a|b⟫ = Tr(a†b) is 2^(−L) times the Euclidean product of the coordinates; the factor drops out of M and L.
# Trace = r_{1…1} = overlap with the all-identity product state.
# H: ../data/hamiltonian_L<L>.csv from the Rust binary `hamiltonian` (random c, seed 1: the realization of `ed`,
# `lindblad_entanglement`, `states`).
#
# Initial states (random product specs, App. C): physical = Σ_k w_k ⊗_j (1 + n_jk·σ)/2 with K = --terms terms,
# w_k > 0, Σ w_k = 1; traceless = difference of two such mixtures. --init random: n_jk uniform in the Bloch ball;
# --init scar: n_jk = n_k unit vectors, so every term is a spin coherent state |n_k⟩⟨n_k|^⊗L in the tower. Both
# backends build the same states from the same seed.
#
# Backends:
#   dense  4^L vectors, propagators e^{𝓛 dz} by matrix exponential (L ≤ 6); also full ED of 𝓛 as the reference.
#   mps    two-site TDVP (ITensorMPS, local Arnoldi exponentials), bond dimension ≤ --chi, cutoff 1e-30 (truncation
#          only through --chi); --n-sub internal steps per δt. --expand-every k > 0: global Krylov basis expansion
#          every k internal steps. Without it the first steps are wrong at the 1e-2 level even at full χ (L = 6):
#          the range-4 terms of H lie outside the two-site tangent space while the bonds are small. Skipped once
#          every bond has the full dimension, where TDVP is exact. L_ij = ⟪ρ_i|𝓛ρ_j⟫ with 𝓛ρ_j stored as an MPS
#          of bond dimension ≤ --chi-apply (exact at the full dimension 4^(L/2)). --check-dense 1 (L ≤ 6) also runs
#          the dense backend and prints the largest differences of M and L.
#
# Usage:
#   julia --project=. -t 16 clik.jl --L 6 --gamma 0.01 --backend dense [--open 0|1] [--init random|scar] [--T 300]
#         [--dt 0.5] [--alpha 0.0005] [--seed 1] [--terms 4] [--eps 1e-12] [--n-T 10]
#         mps only: [--chi 64] [--chi-apply 4χ] [--n-sub 10] [--expand-every 0] [--check-dense 0]
#
# --open 1: open chain (hamiltonian_open_L<L>.csv). --init scar: initial states built from spin coherent states (tower).
#
# Output (../data, or $SCARS_DATA_DIR), tag = clik_<backend>_L<L>_g<γ>[_open][_scar][_chi<χ>] (χ for mps only):
#   <tag>_eigs.csv    T, n_states, rank, re, im: all Krylov eigenvalues of the block |z| ≤ T, for the n-T nested
#                     T = T_max/n-T, 2 T_max/n-T, …, T_max (rank = Gram eigenvalues kept)
#   <tag>_states.csv  n, label (0 physical, 1 traceless), z_re, z_im, trace_re, trace_im, bond_dim, seconds
#                     (trace of the normalized stored state; bond_dim 0 for dense)
#   clik_ed_L<L>_g<γ>[_open].csv   re, im: all eigenvalues of 𝓛 (dense backend), compared with the Rust ED
#                     (used_data/lindblad_entanglement_L<L>_g<γ>.csv, or spectrum_open_L<L>.csv) if that file exists.

include("lindblad_tdvp.jl")
using Random

# ---- input --------------------------------------------------------------------------------------

const DATA = get(ENV, "SCARS_DATA_DIR", joinpath(@__DIR__, "..", "data"))

"H from hamiltonian_L<L>.csv (hamiltonian_open_L<L>.csv for the open chain), in the format of `hamiltonian_terms` (sorted site => axis list → coefficient)."
function read_hamiltonian(L; open_chain = false)
    name = open_chain ? "hamiltonian_open_L$(L).csv" : "hamiltonian_L$(L).csv"
    path = joinpath(DATA, name)
    isfile(path) || error("$path not found: run `cargo run --release --bin hamiltonian -- --L $L --open $(Int(open_chain))` in rust/")
    terms = Dict{Vector{Pair{Int, Int}}, Float64}()
    for line in Iterators.drop(eachline(path), 1)
        string, coef = split(line, ",")
        length(string) == L || error("$path: string $string is not of length $L")
        key = [j => findfirst(string[j], "XYZ") for j in 1:L if string[j] != 'I']
        terms[key] = parse(Float64, coef)
    end
    return terms
end

"--key value options; unknown keys are an error."
function parse_options(args, defaults)
    opts = copy(defaults)
    iseven(length(args)) || error("options are given as --key value")
    for (key, value) in Iterators.partition(args, 2)
        startswith(key, "--") || error("expected --key, got $key")
        name = key[3:end]
        haskey(opts, name) || error("unknown option $key; accepted: --" * join(sort(collect(keys(opts))), ", --"))
        opts[name] = value
    end
    return opts
end

# ---- initial states (product specs) ---------------------------------------------------------------

"""
Spec: list of (coefficient, per-site Pauli coordinates (1, n_x, n_y, n_z)) terms; a mixture of K product states
with random weights. scar = false: independent n on every site, uniform in the Bloch ball. scar = true: spin
coherent states, one unit vector n for all sites, ⊗_j (1 + n·σ_j)/2 = |n⟩⟨n|^⊗L, which lies in the tower
(P ρ P = ρ); mixtures of them span all tower operators |Φ_m⟩⟨Φ_n|.
"""
function random_mixture(rng, L, K; scar = false)
    w = rand(rng, K)
    w ./= sum(w)
    spec = Tuple{Float64, Vector{Vector{Float64}}}[]
    for k in 1:K
        if scar
            n = randn(rng, 3)
            vecs = fill([1.0; n / norm(n)], L)
        else
            vecs = map(1:L) do _
                n = randn(rng, 3)
                n .*= rand(rng)^(1 / 3) / norm(n)      # uniform in the Bloch ball
                [1.0; n]
            end
        end
        push!(spec, (w[k], vecs))
    end
    return spec
end

physical_spec(rng, L, K; scar = false) = random_mixture(rng, L, K; scar)
function traceless_spec(rng, L, K; scar = false)
    a, b = random_mixture(rng, L, K; scar), random_mixture(rng, L, K; scar)
    return [a; [(-c, v) for (c, v) in b]]
end

"Product MPS ⊗_j vecs[j] with explicit links of dimension 1."
function product_mps(sites, vecs)
    L = length(sites)
    links = [Index(1, "Link,l=$j") for j in 1:(L - 1)]
    ψ = MPS(L)
    for j in 1:L
        T = ITensor(ComplexF64.(vecs[j]), sites[j])
        j > 1 && (T *= onehot(links[j - 1] => 1))
        j < L && (T *= onehot(links[j] => 1))
        ψ[j] = T
    end
    return ψ
end

spec_mps(sites, spec) = +([c * product_mps(sites, v) for (c, v) in spec]...; alg = "directsum")

"Dense vector of an MPS, site 1 fastest (the ordering of the dense 𝓛 below)."
todense(ψ::MPS, sites) = vec(Array(prod(ψ), sites...))

# ---- backends -----------------------------------------------------------------------------------

abstract type Backend end

struct Dense <: Backend
    G::Matrix{Float64}
    one::Vector{ComplexF64}
    props::Dict{ComplexF64, Matrix{ComplexF64}}
end

function Dense(mpo::MPO, sites)
    D = 4^length(sites)
    G = reshape(Array(prod(mpo), prime.(sites)..., sites...), D, D)
    one = zeros(ComplexF64, D)
    one[1] = 1                                          # all-identity string: first index
    return Dense(G, one, Dict{ComplexF64, Matrix{ComplexF64}}())
end

function evolve(b::Dense, ψ, dz)
    P = get!(() -> exp(ComplexF64.(b.G) .* dz), b.props, dz)
    return P * ψ
end
trace(b::Dense, ψ) = dot(b.one, ψ)
bond_dim(::Dense, ψ) = 0

"Stored states and 𝓛 applied to them (dense), as the first n columns."
mutable struct DenseStore
    states::Matrix{ComplexF64}
    gstates::Matrix{ComplexF64}
    n::Int
end
store(b::Dense) = DenseStore(zeros(ComplexF64, size(b.G, 1), 0), zeros(ComplexF64, size(b.G, 1), 0), 0)

"Appends ψ; returns ⟪ρ_i|ψ⟫, ⟪ρ_i|𝓛|ψ⟫, ⟪ψ|𝓛|ρ_i⟫ for every stored ρ_i (ψ included, last)."
function add!(b::Dense, s::DenseStore, ψ)
    if s.n == size(s.states, 2)                         # grow the storage by doubling
        grow(A) = [A zeros(ComplexF64, size(A, 1), max(16, size(A, 2)))]
        s.states, s.gstates = grow(s.states), grow(s.gstates)
    end
    s.n += 1
    s.states[:, s.n] = ψ
    s.gstates[:, s.n] = b.G * ψ
    S, GS = view(s.states, :, 1:s.n), view(s.gstates, :, 1:s.n)
    return S' * ψ, S' * GS[:, s.n], conj.(GS' * ψ)
end

struct MPSBackend <: Backend
    G::MPO
    one::MPS
    chi::Int
    chi_apply::Int
    n_sub::Int
    expand_every::Int
end

"Every bond at the dimension of the full space (then two-site TDVP is exact and expansion adds nothing)."
function full_bonds(ψ::MPS)
    L = length(ψ)
    return all(linkdim(ψ, j) == min(4^j, 4^(L - j)) for j in 1:(L - 1))
end

function evolve(b::MPSBackend, ψ, dz)
    for step in 1:b.n_sub
        if b.expand_every > 0 && (step - 1) % b.expand_every == 0 && !full_bonds(ψ)
            ψ = expand(ψ, b.G; alg = "global_krylov", krylovdim = 2, cutoff = 1.0e-12,
                apply_kwargs = (; maxdim = b.chi, cutoff = 1.0e-12))
        end
        ψ = tdvp(b.G, dz / b.n_sub, ψ;
            nsteps = 1, nsite = 2, maxdim = b.chi, cutoff = 1.0e-30, normalize = false, outputlevel = 0,
            updater_kwargs = (; ishermitian = false, issymmetric = false, tol = 1.0e-12, krylovdim = 30, eager = true))
    end
    return ψ
end
trace(b::MPSBackend, ψ) = inner(b.one, ψ)
bond_dim(::MPSBackend, ψ) = maxlinkdim(ψ)

"Stored states and 𝓛 applied to them, compressed to bond dimension ≤ chi_apply (MPS)."
struct MPSStore
    states::Vector{MPS}
    gstates::Vector{MPS}
end
store(::MPSBackend) = MPSStore(MPS[], MPS[])

function add!(b::MPSBackend, s::MPSStore, ψ)
    push!(s.states, ψ)
    push!(s.gstates, apply(b.G, ψ; maxdim = b.chi_apply, cutoff = 1.0e-30))
    n = length(s.states)
    m, lcol, lrow = zeros(ComplexF64, n), zeros(ComplexF64, n), zeros(ComplexF64, n)
    blas = BLAS.get_num_threads()
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)           # threads over the stored states instead
    Threads.@threads for i in 1:n
        m[i] = inner(s.states[i], ψ)
        lcol[i] = inner(s.states[i], s.gstates[n])
        lrow[i] = inner(ψ, s.gstates[i])
    end
    BLAS.set_num_threads(blas)
    return m, lcol, lrow
end

# ---- CLIK (Alg. 1) --------------------------------------------------------------------------------

"Runs both evolutions, returns M, L, labels, z and per-state traces, bond dimensions and times."
function run_clik(b::Backend, ψp, ψu; dt, T, α, log_prefix = "")
    n_steps = round(Int, T / dt)
    N = 2 * (n_steps + 1)
    M, Lm = zeros(ComplexF64, N, N), zeros(ComplexF64, N, N)
    labels, z, tr, bonds, secs = Int[], ComplexF64[], ComplexF64[], Int[], Float64[]
    s = store(b)
    dzs = (dt, dt * exp(-im * α))
    states = [ψp / norm(ψp), ψu / norm(ψu)]
    start = time()
    for step in 0:n_steps
        for e in 1:2
            if step > 0
                states[e] = evolve(b, states[e], dzs[e])
                states[e] /= norm(states[e])
            end
            m, lcol, lrow = add!(b, s, states[e])
            n = length(m)
            M[1:n, n] = m
            M[n, 1:n] = conj.(m)
            Lm[1:n, n] = lcol
            Lm[n, 1:n] = lrow
            push!(labels, e - 1)
            push!(z, step * dzs[e])
            push!(tr, trace(b, states[e]))
            push!(bonds, bond_dim(b, states[e]))
            push!(secs, time() - start)
        end
        if step % max(1, n_steps ÷ 20) == 0
            @printf("%s t = %.2f: %d states, bond dims %d, %d, |tr u| = %.1e, %.1f s\n", log_prefix, step * dt,
                length(labels), bonds[end - 1], bonds[end], abs(tr[end]), time() - start)
            flush(stdout)
        end
    end
    return (; M, Lm, labels, z, tr, bonds, secs)
end

"Alg. 2: eigenvalues of 𝓛 projected on the span of the states with Gram matrix M; sorted by Re λ descending."
function krylov_eigs(M, Lm; eps = 1.0e-12)
    d = sqrt.(real.(diag(M)))
    F = eigen(Hermitian(M ./ (d * d')))
    keep = F.values .> eps
    X = F.vectors[:, keep] ./ sqrt.(F.values[keep])' ./ d
    λ = eigvals(X' * Lm * X)
    return sort(λ; by = x -> (-real(x), -imag(x))), count(keep)
end

# ---- exact reference ------------------------------------------------------------------------------

function exact_spectrum(b::Dense, L, γ, open_chain)
    t0 = time()
    λ = eigvals(b.G)
    path = joinpath(DATA, "clik_ed_L$(L)_g$(γ)$(open_chain ? "_open" : "").csv")
    open(path, "w") do io
        println(io, "re,im")
        for x in λ
            @printf(io, "%.15e,%.15e\n", real(x), imag(x))
        end
    end
    up = filter(x -> imag(x) > 1.0e-9, λ)
    αmax = minimum(atan(-real(x) / imag(x)) for x in up)
    @printf("ED: %d eigenvalues in %.1f s, slowest nonzero Re λ = %.6f, α_max = %.5f -> %s\n", length(λ),
        time() - t0, maximum(real(x) for x in λ if abs(x) > 1.0e-9), αmax, path)
    # Rust ED: spectrum_open_L<L>.csv (`ed --open 1`, rows of this γ) or lindblad_entanglement_L<L>_g<γ>.csv
    rust = open_chain ? joinpath(DATA, "spectrum_open_L$(L).csv") :
           joinpath(DATA, "used_data", "lindblad_entanglement_L$(L)_g$(γ).csv")
    if isfile(rust)
        cols = split(first(eachline(rust)), ",")
        ire, iim, ig = findfirst(==("re"), cols), findfirst(==("im"), cols), findfirst(==("gamma"), cols)
        rows = [split(l, ",") for l in Iterators.drop(eachline(rust), 1)]
        filter!(r -> parse(Float64, r[ig]) == γ, rows)
        ref = [parse(Float64, r[ire]) + im * parse(Float64, r[iim]) for r in rows]
        dist = maximum(minimum(abs.(λ .- r)) for r in ref)
        @printf("ED check against %s: %d eigenvalues, largest distance to the Julia spectrum %.2e\n",
            rust, length(ref), dist)
    end
    return λ
end

# ---- main ---------------------------------------------------------------------------------------

function main(args)
    o = parse_options(args, Dict("L" => "6", "gamma" => "0.01", "backend" => "dense", "T" => "300", "dt" => "0.5",
        "alpha" => "0.0005", "seed" => "1", "terms" => "4", "eps" => "1e-12", "n-T" => "10", "chi" => "64",
        "n-sub" => "10", "expand-every" => "0", "chi-apply" => "0", "check-dense" => "0", "open" => "0",
        "init" => "random"))
    L, γ = parse(Int, o["L"]), parse(Float64, o["gamma"])
    T, dt, α = parse(Float64, o["T"]), parse(Float64, o["dt"]), parse(Float64, o["alpha"])
    K, ε, nT = parse(Int, o["terms"]), parse(Float64, o["eps"]), parse(Int, o["n-T"])
    χ = parse(Int, o["chi"])
    open_chain = o["open"] == "1"
    o["init"] in ("random", "scar") || error("--init random or scar")
    scar = o["init"] == "scar"
    # tag: clik_<backend>_L<L>_g<γ>[_open][_scar][_chi<χ>]
    label = "L$(L)_g$(γ)" * (open_chain ? "_open" : "") * (scar ? "_scar" : "")

    sites = siteinds("PauliVec", L)
    G = MPO(lindbladian(L, γ; terms = read_hamiltonian(L; open_chain)), sites)
    rng = Xoshiro(parse(Int, o["seed"]))
    specp, specu = physical_spec(rng, L, K; scar), traceless_spec(rng, L, K; scar)
    mps_p, mps_u = spec_mps(sites, specp), spec_mps(sites, specu)
    one = product_mps(sites, fill([1.0, 0, 0, 0], L))
    @printf("L = %d (%s), γ = %g, %s backend, %s initial states, T = %g, δt = %g, α = %g, %d threads, MPO bond dimension %d\n",
        L, open_chain ? "open" : "periodic", γ, o["backend"], o["init"], T, dt, α, Threads.nthreads(), maxlinkdim(G))
    @printf("initial traces: physical %.3e, traceless %.3e\n", abs(inner(one, mps_p)), abs(inner(one, mps_u)))

    if o["backend"] == "dense"
        L <= 6 || error("dense backend: L ≤ 6")
        b = Dense(G, sites)
        exact_spectrum(b, L, γ, open_chain)
        res = run_clik(b, todense(mps_p, sites), todense(mps_u, sites); dt, T, α, log_prefix = "dense")
        tag = "clik_dense_" * label
    elseif o["backend"] == "mps"
        χa = o["chi-apply"] == "0" ? 4χ : parse(Int, o["chi-apply"])
        b = MPSBackend(G, one, χ, χa, parse(Int, o["n-sub"]), parse(Int, o["expand-every"]))
        res = run_clik(b, mps_p, mps_u; dt, T, α, log_prefix = "mps")
        tag = "clik_mps_" * label * "_chi$(χ)"
        if o["check-dense"] == "1"
            L <= 6 || error("--check-dense: L ≤ 6")
            ref = run_clik(Dense(G, sites), todense(mps_p, sites), todense(mps_u, sites); dt, T, α, log_prefix = "dense")
            for n in (2, length(ref.labels) ÷ 4, length(ref.labels) ÷ 2, length(ref.labels))
                @printf("check-dense, first %d states: max |ΔM| = %.2e, max |ΔL| = %.2e\n", n,
                    maximum(abs.(res.M[1:n, 1:n] - ref.M[1:n, 1:n])), maximum(abs.(res.Lm[1:n, 1:n] - ref.Lm[1:n, 1:n])))
            end
        end
    else
        error("--backend dense or mps")
    end

    open(joinpath(DATA, tag * "_states.csv"), "w") do io
        println(io, "n,label,z_re,z_im,trace_re,trace_im,bond_dim,seconds")
        for n in eachindex(res.labels)
            @printf(io, "%d,%d,%.10f,%.10f,%.12e,%.12e,%d,%.3f\n", n, res.labels[n], real(res.z[n]), imag(res.z[n]),
                real(res.tr[n]), imag(res.tr[n]), res.bonds[n], res.secs[n])
        end
    end
    path = joinpath(DATA, tag * "_eigs.csv")
    open(path, "w") do io
        println(io, "T,n_states,rank,re,im")
        for k in 1:nT
            Tk = T * k / nT
            n = count(x -> abs(x) <= Tk + 1.0e-9, res.z)
            λ, rank = krylov_eigs(res.M[1:n, 1:n], res.Lm[1:n, 1:n]; eps = ε)
            for x in λ
                @printf(io, "%g,%d,%d,%.15e,%.15e\n", Tk, n, rank, real(x), imag(x))
            end
            @printf("T = %g: %d states, rank %d, slowest: %s\n", Tk, n, rank,
                join([@sprintf("%.5f%+.5fi", real(x), imag(x)) for x in λ[1:min(6, end)]], ", "))
        end
    end
    println("-> ", path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
