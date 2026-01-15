abstract type BoundedSubproblemCache end

mutable struct ColemanandLiCache{S<:SubProblemStrategy,F,D,JV,V,M,T} <: BoundedSubproblemCache
    factorization::F
    scaling_matrix::D
    Jv::JV
    v::V
    # Workspaces (matching SubproblemCache)
    J_buffer::M              
    p::Vector{T}             # Step direction 
    p_newton::Vector{T}      # Gradient dp/dλ
    R_buffer::Matrix{T}      # n x n mutable R
    rhs_buffer::Vector{T}    # n mutable RHS
    qtf_buffer::Vector{T}    # Stores Q'f
    v_row::Vector{T}         # Row workspace
    perm_buffer::Vector{T}   # Buffer for permutation operations

    ColemanandLiCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:SubProblemStrategy, Sc<:BoundedScalingStrategy, T} = begin
        m, n = size(J)
        F = factorize(strategy, J)
        Dk, A, v = scaling(scaling_strat, J; kwargs...)
        
        # Initialize workspaces
        if J isa StridedMatrix{T}
            J_buffer = similar(J)
        else
            J_buffer = nothing
        end
        
        p = zeros(T, n)
        p_newton = zeros(T, n)
        R_buffer = zeros(T, n, n)
        rhs_buffer = zeros(T, max(m, n)) 
        qtf_buffer = zeros(T, max(m, n))
        v_row = zeros(T, n)
        perm_buffer = zeros(T, n)

        new{S,typeof(F),typeof(Dk), typeof(A), typeof(v), typeof(J_buffer), T}(
            F, Dk, A, v, J_buffer, 
            p, p_newton, R_buffer, rhs_buffer, qtf_buffer, v_row, perm_buffer
        )
    end
end


"""
    solve_subproblem(strategy::SubProblemStrategy, J, f, radius, cache::ColemanandLiCache)

Solve the trust region subproblem for bounded problems.
"""
function solve_subproblem(
    strategy::SubProblemStrategy,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::ColemanandLiCache,
) where {T<:Real}
    Dk = cache.scaling_matrix
    A = cache.Jv
    
    n = size(J, 2)
    # We can use cache.p to store δgn initially
    δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation # Allocating for now
    
    if norm(Dk*δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        λ, δ = find_λ_colemanandli(strategy, cache, radius, J, Dk, A, f, 200, 1e-6)
    end
    return λ, δ
end

"""
    find_λ_colemanandli(strategy::QRrecursiveSolve, F, Δ, J, D,A, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using Recursive QR.
"""
function find_λ_colemanandli(strategy::QRrecursiveSolve, cache, Δ, J, D, A, f, maxiters, θ = 1e-4)
    # Unpack buffers
    F = cache.factorization
    p = cache.p
    dpdλ = cache.p_newton
    R_buffer = cache.R_buffer
    rhs_buffer = cache.rhs_buffer
    qtf = cache.qtf_buffer
    v_row = cache.v_row
    perm_buffer = cache.perm_buffer
    
    m, n = size(J)
    
    # FIX 1: Type Stable Permutation
    # ensure perm is always Vector{Int}
    perm = hasproperty(F, :p) ? F.p : collect(1:n)
    
    # Lambda initialization
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀

    # 1. Setup (One-time allocation for Q'*f allowed)
    qtf_src = F.Q' * f 
    @inbounds for i in 1:m
        qtf[i] = qtf_src[i]
    end
    
    # 2. Main Loop
    for i in 1:maxiters
        # FIX 3: Efficient R copy
        # Access F.factors directly if possible to avoid UpperTriangular wrapper allocs
        # If F is QRPivoted or QR, it has :factors.
        src_R = hasproperty(F, :factors) ? F.factors : F.R
        
        if m >= n
             @inbounds for j in 1:n, k in 1:j; R_buffer[k,j] = src_R[k,j]; end
        else
             fill!(R_buffer, 0.0)
             @inbounds for j in 1:n, k in 1:min(j,m); R_buffer[k,j] = src_R[k,j]; end
        end
        
        # Reset RHS
        @inbounds for k in 1:n
            rhs_buffer[k] = -qtf[k]
        end

        # Recursive Update
        solve_damped_system_recursive_coleman!(p, R_buffer, rhs_buffer, λ, n, D, A, perm, v_row)
        
        norm_Dp = norm(D*p)
        if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
            break
        end
        
        ϕ = norm_Dp - Δ
        if ϕ < 0; uₖ = λ; else; lₖ = λ; end
        
        # Derivative
        solve_for_dp_dlambda_scaled!(dpdλ, R_buffer, p, D, perm, perm_buffer)
        
        # Newton Update
        denominator = (D*p)' * (D * dpdλ)
        λ = λ - (norm_Dp - Δ)/Δ * ( (norm_Dp^2) / denominator )
        
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ + 0.01*(uₖ - lₖ), √(lₖ * uₖ))
        end
    end
    return λ, p
end


function solve_damped_system_recursive_coleman!(p_cache, R_cache, QTr_cache, λ, n, D, A, perm, v_row)
    sqrt_λ = √λ

    for c_idx in 1:n
        fill!(v_row, 0.0)
        
        var_idx = perm[c_idx]
        d_val = D isa Diagonal ? D[var_idx, var_idx] : D[var_idx]
        a_val = A isa Diagonal ? A[var_idx, var_idx] : A[var_idx]
        v_row[c_idx] = sqrt_λ * d_val + √a_val
        v_rhs = 0.0
        
        for i in c_idx:n
            r_ii = R_cache[i, i]
            v_val = v_row[i]
            
            if abs(v_val) > 0 || i == c_idx
                c, s = compute_givens(r_ii, v_val)
                
                R_cache[i, i] = c * r_ii + s * v_val
                
                @inbounds for k in (i + 1):n
                    val_R = R_cache[i, k]
                    val_v = v_row[k]
                    R_cache[i, k] = c * val_R + s * val_v
                    v_row[k]      = -s * val_R + c * val_v
                end
                
                val_rhs_R = QTr_cache[i]
                QTr_cache[i] = c * val_rhs_R + s * v_rhs
                v_rhs        = -s * val_rhs_R + c * v_rhs
            end
        end
    end
    
    # Use View for RHS to match dimension n
    ldiv!(p_cache, UpperTriangular(R_cache), view(QTr_cache, 1:n))
    
    copyto!(v_row, p_cache)
    @inbounds for i in 1:n
        p_cache[perm[i]] = v_row[i]
    end
    
    return p_cache
end