abstract type BoundedSubproblemCache end

mutable struct ColemanandLiCache{S<:SubProblemStrategy,F,D,JV,V,Vec,Mat} <: BoundedSubproblemCache
    factorization::F
    scaling_matrix::D
    Jv::JV
    v::V
    # Workspaces (matching SubproblemCache)
    p::Vec
    p_tmp::Vec
    b_aug::Vec
    J_aug::Mat
    # For Recursive Solver
    R_buffer::Mat
    rhs_buffer::Vec
    v_row::Vec
    rhs_orig::Vec

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
        p = zeros(T, n)
        p_tmp = zeros(T, n)
        
        J_aug = zeros(T, m + n, n)
        b_aug = zeros(T, m + n)
        
        R_buffer = zeros(T, n, n)
        rhs_buffer = zeros(T, n)
        v_row = zeros(T, n)
        rhs_orig = zeros(T, n)

        new{S,typeof(F),typeof(Dk), typeof(A), typeof(v), Vector{T}, Matrix{T}}(
            F, Dk, A, v, p, p_tmp, b_aug, J_aug, R_buffer, rhs_buffer, v_row, rhs_orig
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
    F = cache.factorization
    Dk = cache.scaling_matrix
    A = cache.Jv
    
    # We can use cache.p to store δgn initially
    δgn = F \ -f # Allocating for now
    
    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        D = √A * Dk # This allocates a new diagonal matrix?
        # A and Dk are diagonal.
        # Ideally we'd compute this lazily or in a buffer.
        # But find_λ_scaled takes D.
        # Let's keep it allocating for now or optimize later.
        
        λ, δ = find_λ_scaled(strategy, F, radius, J, D, f, cache, 200, 1e-6)
    end
    return λ, δ
end