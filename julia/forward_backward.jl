# Forward/backward (meet-in-the-middle) evolution: expectation values of the Lindblad dynamics from a state evolved
# forward and an observable evolved backward (Heisenberg picture), each over half of the time.
#
#     dρ/dt = 𝓛ρ = −i[H, ρ] + 𝓓(ρ),   𝓓 one of the single-site channels of rust/src/noise.rs
#     ⟨O⟩(t₁ + t₂) = Tr[O e^{𝓛(t₁+t₂)} ρ₀] = Tr[(e^{𝓛† t₂} O) (e^{𝓛 t₁} ρ₀)]
#
# Coordinates as in lindblad_tdvp.jl: ρ = 2^(−L) Σ_P r_P P, r_P = Tr(ρP), with 𝓛 the real matrix G (dr/dt = G r).
# An observable O = Σ_P o_P P, o_P = Tr(OP)/2^L, has Tr(Oρ) = Σ_P o_P r_P = o·r, and d(o·r)/dt = o·Gr = (Gᵀo)·r, so
# the backward (Heisenberg) evolution is do/dt = Gᵀ o. Per site, A^a is symmetric and B^a antisymmetric, and every
# Hamiltonian term carries an odd number of B factors, so the Hamiltonian part of Gᵀ is minus that of G (H → −H).
# The dissipator is diagonal for the Hermitian channels (Gᵀ = G there); decay and thermal have the feed σᶻ → −𝟙
# (dr_z/dt ∋ −γ r_𝟙), whose transpose maps 𝟙 → σᶻ. 𝓛†(𝟙) = 0 for every channel, so Tr ρ(t) = 𝟙·r(t) exactly and
# ⟨O⟩ is divided by the trace of the forward state (TDVP truncation does not keep it).
#
# Both halves are MPS of the 4-dimensional PauliVec sites (bond dimension ≤ --chi). Stepping (`step`): a Taylor series
# of e^{Gτ} with exact MPO applications while the bond dimension is below --chi-switch, two-site TDVP with global
# Krylov expansion above (--method hybrid, default); --method taylor or tdvp use one of them throughout. TDVP alone
# is wrong at low χ (from a product state), Taylor alone is slow at large χ.
# The initial state is a product (bond dimension 1), an observable (1/L) Σ_j c_j σ^α_j has bond dimension 2; both stay
# simpler than ρ(t) for half the time.
#
# For every output time t_n = n δt, with forward states ρ_i (i ≤ ⌈N/2⌉) and backward observables o_j (j ≤ ⌈N/2⌉):
#   mid     split i = ⌈n/2⌉, j = n − i
#   fwd     forward only (j = 0): ρ evolved to t_n, same χ           (with --full 1)
#   bwd     backward only (i = 0): O evolved (Heisenberg) to t_n     (with --full 1)
#   spread  max − min of ⟨O⟩ over all stored splits i + j = n (a truncation-error estimate; 0 when exact)
#
# Observables (--obs, any of): mx, my, mz (uniform, (1/L) Σ_j σ^α_j), sx, sy, sz (staggered, (1/L) Σ_j (−1)^j σ^α_j,
# j = 0 … L−1), the columns of the Rust `states` output. Initial states (--state): minus-x |−⟩^⊗L, neel-x |+−+−…⟩,
# neel |↑↓↑↓…⟩ (spin 0 up / along +x, as in states.rs).
#
# H: ../data/hamiltonian_L<L>.csv (random c, seed 1, the realization of `states`; `cargo run --release --bin
# hamiltonian -- --L <L>`).
#
# Usage:
#   julia --project=. -t 8 forward_backward.jl --L 8 --gamma 0.1 --state minus-x [--channel z] [--nbar 1] [--chi 64]
#         [--t-max 10] [--dt 0.1] [--n-sub 2] [--method hybrid|taylor|tdvp] [--chi-switch 16] [--cutoff 1e-14]
#         [--obs mx,my,mz,sx,sy,sz]
#         [--full 1] [--check 0] [--ref path.csv]
#   --check 1 (L ≤ 6): dense G and Gᵀ from the two MPOs, |G_mpoᵀ − G_adjoint| and the trace column.
#   --ref: Rust `states` CSV to compare with (default: ../data/used_data/states_<state>_L<L>_g<γ><channel>.csv if it
#          exists); prints the largest deviation of fwd, mid and bwd per observable.
#
# Output (../data, or $SCARS_DATA_DIR): fb_<state>_L<L>_g<γ><channel>_chi<χ>.csv with columns
#   L, gamma, chi, t, obs, fwd, mid, bwd, spread, n_splits, bond_fwd, bond_bwd, seconds
# (fwd, bwd empty with --full 0; bond_fwd, bond_bwd: bond dimensions of the two halves of the mid split).

include("clik.jl")

# ---- channels ------------------------------------------------------------------------------------

# projectors on the coordinates of σˣ, σʸ, σᶻ, and the feed (row σᶻ, column 𝟙) and its transpose
ITensors.op(::OpName"Px", ::SiteType"PauliVec") = [0.0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
ITensors.op(::OpName"Py", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 0]
ITensors.op(::OpName"Pz", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
ITensors.op(::OpName"Feed", ::SiteType"PauliVec") = [0.0 0 0 0; 0 0 0 0; 0 0 0 0; 1 0 0 0]
ITensors.op(::OpName"FeedT", ::SiteType"PauliVec") = [0.0 0 0 1; 0 0 0 0; 0 0 0 0; 0 0 0 0]

"Per unit of γ: diagonal rates of σˣ, σʸ, σᶻ and the feed coefficient (rust/src/noise.rs, `site_diag`, `feed`)."
function channel_rates(ch, nbar)
    ch == "z" && return (-2.0, -2.0, 0.0), 0.0
    ch == "x" && return (0.0, -2.0, -2.0), 0.0
    ch == "y" && return (-2.0, 0.0, -2.0), 0.0
    ch == "dep" && return (-2.0, -2.0, -2.0), 0.0
    ch == "decay" && return (-0.5, -0.5, -1.0), -1.0
    if ch == "thermal"
        d = 2nbar + 1
        return (-d / 2, -d / 2, -d), -1.0
    end
    error("--channel z, x, y, dep, decay or thermal, got $ch")
end

"Output name of a number as Rust prints an f64 (1.0 → 1, 0.01 → 0.01)."
num(x) = isinteger(x) ? string(Int(x)) : string(x)
channel_suffix(ch, nbar) = ch == "z" ? "" : ch == "thermal" ? "_chthermal_n$(num(nbar))" : "_ch$(ch)"

"""
G (adjoint = false) or Gᵀ (adjoint = true) as an OpSum: the Hamiltonian part of `lindbladian` (lindblad_tdvp.jl),
with the sign flipped for Gᵀ, plus the channel's dissipator on every site.
"""
function generator(L, γ, ch; terms, nbar = 1.0, adjoint = false)
    os = lindbladian(L, 0.0; terms)                     # Hamiltonian part only (the σᶻ term is added below)
    adjoint && (os = -1.0 * os)
    (dx, dy, dz), feed = channel_rates(ch, nbar)
    for j in 1:L
        dx != 0 && (os += (γ * dx, "Px", j))
        dy != 0 && (os += (γ * dy, "Py", j))
        dz != 0 && (os += (γ * dz, "Pz", j))
        feed != 0 && (os += (γ * feed, adjoint ? "FeedT" : "Feed", j))
    end
    return os
end

# ---- states and observables ----------------------------------------------------------------------

"Coordinates (1, n_x, n_y, n_z) of site j (0-based) of the initial product state."
function initial_site(state, j)
    s = iseven(j) ? 1.0 : -1.0
    state == "minus-x" && return [1.0, -1.0, 0.0, 0.0]
    state == "neel-x" && return [1.0, s, 0.0, 0.0]
    state == "neel" && return [1.0, 0.0, 0.0, s]
    error("--state minus-x, neel-x or neel, got $state")
end

"Coordinate MPS (o_P = Tr(OP)/2^L) of O = Σ_j c[j] σ^α_j, bond dimension 2 (α = 1, 2, 3 for x, y, z)."
function sum_mps(sites, α, c)
    L = length(sites)
    links = [Index(2, "Link,l=$j") for j in 1:(L - 1)]
    ψ = MPS(L)
    for j in 1:L
        if j == 1
            A = zeros(4, 2)
            A[1, 1] = 1
            A[α + 1, 2] = c[1]
            ψ[j] = ITensor(A, sites[j], links[j])
        elseif j == L
            A = zeros(2, 4)
            A[1, α + 1] = c[L]
            A[2, 1] = 1
            ψ[j] = ITensor(A, links[j - 1], sites[j])
        else
            A = zeros(2, 4, 2)
            A[1, 1, 1] = 1
            A[1, α + 1, 2] = c[j]
            A[2, 1, 2] = 1
            ψ[j] = ITensor(A, links[j - 1], sites[j], links[j])
        end
    end
    return ψ
end

function observable(sites, name)
    L = length(sites)
    α = findfirst(name[2], "xyz")
    (length(name) == 2 && name[1] in "ms" && α !== nothing) || error("observable mx, my, mz, sx, sy or sz, got $name")
    c = name[1] == 'm' ? fill(1.0 / L, L) : [(-1.0)^(j - 1) / L for j in 1:L]
    return sum_mps(sites, α, c)
end

# ---- evolution -----------------------------------------------------------------------------------

"""
One output step δt of e^{G δt} ψ in n substeps, n = max(n_sub, ⌈δt γ / 0.05⌉) (the output-step rule of the Rust
binaries, so δt γ ≤ 0.05 per substep).
  taylor  Σ_k (G τ)^k/k! ψ with exact MPO applications (`apply`, density-matrix algorithm; zip-up is wrong by 16 % at
          L = 6), compressed to χ ≤ chi with the SVD cutoff, summed until the new term is below tol relative to the
          sum. No projection error, but ≈ 25 s per substep at L = 6, χ = 64 (20 terms).
  tdvp    two-site TDVP with a global Krylov expansion while bonds are not full. Exact at full bonds, but from a
          product state one step δt = 0.1 is off by ≈ 1e-2 in norm at L = 6 (δt² scaling), with or without expansion:
          the range-4 terms of H leave the two-site tangent space of a low-χ state.
  hybrid  (default) taylor while the bond dimension is below chi_switch, tdvp after.
"""
function step(G::MPO, ψ::MPS, dt; chi, n_sub, γ, method, cutoff, chi_switch = 16, tol = 1.0e-13,
    expand_cutoff = 1.0e-12)
    method in ("taylor", "tdvp", "hybrid") || error("--method taylor, tdvp or hybrid, got $method")
    n = max(n_sub, ceil(Int, dt * γ / 0.05))
    τ = dt / n
    for _ in 1:n
        if method == "taylor" || (method == "hybrid" && maxlinkdim(ψ) < chi_switch)
            term, total = ψ, ψ
            for k in 1:60
                term = (τ / k) * apply(G, term; maxdim = chi, cutoff)
                total = +(total, term; maxdim = chi, cutoff)
                norm(term) < tol * norm(total) && break
                k == 60 && error("Taylor series did not converge in 60 terms (τ = $τ)")
            end
            ψ = total
        else
            if !full_bonds(ψ)
                ψ = expand(ψ, G; alg = "global_krylov", krylovdim = 2, cutoff = expand_cutoff,
                    apply_kwargs = (; maxdim = chi, cutoff = expand_cutoff))
            end
            ψ = tdvp(G, τ, ψ;
                nsteps = 1, nsite = 2, maxdim = chi, cutoff, normalize = false, outputlevel = 0,
                updater_kwargs = (; ishermitian = false, issymmetric = false, tol = 1.0e-12, krylovdim = 30,
                    eager = true))
        end
    end
    return ψ
end

"o·r for two coordinate MPS (real up to round-off; the product states are stored as ComplexF64)."
rinner(a::MPS, b::MPS) = real(inner(a, b))

"Reads a Rust `states` CSV: Dict t (rounded to 1e-6) → Dict observable → value."
function read_reference(path)
    lines = collect(eachline(path))
    head = split(lines[1], ",")
    ref = Dict{Float64, Dict{String, Float64}}()
    for line in lines[2:end]
        f = split(line, ",")
        length(f) == length(head) || continue                       # cut last line
        row = Dict(String(head[k]) => parse(Float64, f[k]) for k in eachindex(f))
        ref[round(row["t"]; digits = 6)] = row
    end
    return ref
end

function fb_main(args)
    o = parse_options(args, Dict("L" => "8", "gamma" => "0.1", "state" => "minus-x", "channel" => "z", "nbar" => "1",
        "chi" => "64", "t-max" => "10", "dt" => "0.1", "n-sub" => "2", "method" => "hybrid", "chi-switch" => "16",
        "obs" => "mx,my,mz,sx,sy,sz", "full" => "1", "check" => "0", "ref" => "", "cutoff" => "1e-14"))
    L, γ, st, ch = parse(Int, o["L"]), parse(Float64, o["gamma"]), o["state"], o["channel"]
    nbar, χ = parse(Float64, o["nbar"]), parse(Int, o["chi"])
    dt, N = parse(Float64, o["dt"]), round(Int, parse(Float64, o["t-max"]) / parse(Float64, o["dt"]))
    n_sub, full = parse(Int, o["n-sub"]), o["full"] == "1"
    names = split(o["obs"], ",")
    kw = (; chi = χ, n_sub, γ, method = o["method"], cutoff = parse(Float64, o["cutoff"]),
        chi_switch = parse(Int, o["chi-switch"]))

    sites = siteinds("PauliVec", L)
    terms = read_hamiltonian(L)
    G = MPO(generator(L, γ, ch; terms, nbar), sites)
    Gt = MPO(generator(L, γ, ch; terms, nbar, adjoint = true), sites)
    one = product_mps(sites, fill([1.0, 0, 0, 0], L))
    ρ0 = product_mps(sites, [initial_site(st, j - 1) for j in 1:L])
    ops = [observable(sites, String(n)) for n in names]
    @printf("L = %d, γ = %g, channel %s, state %s, χ = %d, δt = %g, N = %d, %d threads; MPO bond dimension %d (G), %d (Gᵀ)\n",
        L, γ, ch, st, χ, dt, N, Threads.nthreads(), maxlinkdim(G), maxlinkdim(Gt))

    if o["check"] == "1"
        L <= 6 || error("--check 1: L ≤ 6")
        D = 4^L
        dense(M) = reshape(Array(prod(M), prime.(sites)..., sites...), D, D)
        Gd, Gtd = dense(G), dense(Gt)
        @printf("check: max |Gᵀ − G_adjoint| = %.2e (max |G| = %.2e), max |𝟙ᵀ G| = %.2e (trace preservation)\n",
            maximum(abs.(transpose(Gd) - Gtd)), maximum(abs.(Gd)), maximum(abs.(Gd[1, :])))
    end

    half = cld(N, 2)
    nf = full ? N : half
    start = time()
    # forward: stored for i ≤ half; fwd-only values on the fly up to nf
    fwd = fill(NaN, N + 1, length(ops))
    bwd = fill(NaN, N + 1, length(ops))
    F = MPS[ρ0]
    trF = Float64[rinner(one, ρ0)]
    ρ = ρ0
    for n in 0:nf
        n > 0 && (ρ = step(G, ρ, dt; kw...))
        tr = rinner(one, ρ)
        for k in eachindex(ops)
            fwd[n + 1, k] = rinner(ops[k], ρ) / tr
        end
        if 0 < n <= half
            push!(F, ρ)
            push!(trF, tr)
        end
        n % max(1, nf ÷ 10) == 0 && @printf("forward  t = %7.3f  χ = %4d  trace %.3e  %.1f s\n", n * dt,
            maxlinkdim(ρ), tr, time() - start)
    end
    # backward: o_j for every observable, stored for j ≤ half; bwd-only values (against ρ₀) up to nf
    B = [MPS[op] for op in ops]
    for k in eachindex(ops)
        ok = ops[k]
        bwd[1, k] = rinner(ok, ρ0) / trF[1]
        for n in 1:nf
            ok = step(Gt, ok, dt; kw...)
            bwd[n + 1, k] = rinner(ok, ρ0) / trF[1]
            n <= half && push!(B[k], ok)
        end
        @printf("backward %s: t = %g, χ = %d, %.1f s\n", names[k], nf * dt, maxlinkdim(ok), time() - start)
    end

    tag = "fb_$(st)_L$(L)_g$(num(γ))$(channel_suffix(ch, nbar))_chi$(χ)"
    path = joinpath(DATA, tag * ".csv")
    mid = fill(NaN, N + 1, length(ops))
    open(path, "w") do io
        println(io, "L,gamma,chi,t,obs,fwd,mid,bwd,spread,n_splits,bond_fwd,bond_bwd,seconds")
        for n in 0:N, k in eachindex(ops)
            splits = [i for i in max(0, n - half):min(n, half)]
            vals = [rinner(B[k][n - i + 1], F[i + 1]) / trF[i + 1] for i in splits]
            i = cld(n, 2)
            mid[n + 1, k] = rinner(B[k][n - i + 1], F[i + 1]) / trF[i + 1]
            f(x) = isnan(x) ? "" : @sprintf("%.12e", x)
            @printf(io, "%d,%s,%d,%.6f,%s,%s,%.12e,%s,%.3e,%d,%d,%d,%.2f\n", L, num(γ), χ, n * dt, names[k],
                f(fwd[n + 1, k]), mid[n + 1, k], f(bwd[n + 1, k]), maximum(vals) - minimum(vals), length(vals),
                maxlinkdim(F[i + 1]), maxlinkdim(B[k][n - i + 1]), time() - start)
        end
    end
    println("-> ", path)

    ref = o["ref"]
    if isempty(ref)
        ref = joinpath(DATA, "used_data", "states_$(st)_L$(L)_g$(num(γ))$(channel_suffix(ch, nbar)).csv")
        isfile(ref) || return
    end
    r = read_reference(ref)
    println("against ", ref, " (max |Δ| over the common times):")
    for k in eachindex(ops)
        dev = Dict(:fwd => 0.0, :mid => 0.0, :bwd => 0.0)
        cnt = 0
        for n in 0:N
            row = get(r, round(n * dt; digits = 6), nothing)
            row === nothing && continue
            cnt += 1
            x = row[String(names[k])]
            dev[:mid] = max(dev[:mid], abs(mid[n + 1, k] - x))
            isnan(fwd[n + 1, k]) || (dev[:fwd] = max(dev[:fwd], abs(fwd[n + 1, k] - x)))
            isnan(bwd[n + 1, k]) || (dev[:bwd] = max(dev[:bwd], abs(bwd[n + 1, k] - x)))
        end
        @printf("  %s (%d times): fwd %.2e, mid %.2e, bwd %.2e\n", names[k], cnt, dev[:fwd], dev[:mid], dev[:bwd])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    fb_main(ARGS)
end
