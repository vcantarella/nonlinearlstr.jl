abstract type BoundedSubproblemCache end
mutable struct ColemandandLiCache{S<:SubProblemStrategy,F,D,JV,V} <: BoundedSubproblemCache
    factorization::F
    scaling_matrix::D
    Jv::JV
    v::V
    ColemanandLiCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix;
        kwargs...,
    ) where {S<:SubProblemStrategy, Sc<:BoundedScalingStrategy} = begin
        F = factorize(strategy, J)
        Dk, A, v = scaling(scaling_strat, J; kwargs)
        new{S,typeof(F),typeof(Dk), typeof(A), typeof(v)}(F, Dk, A, v)
    end
end


"""
    solve_subproblem(strategy::SubProblemStrategy, J::AbstractMatrix{T}, f::AbstractVector{T}, radius::Real, cache) where {T<:Real}

Solve the trust region subproblem to find the optimal step direction and Lagrange multiplier.

The subproblem being solved is:
    min_{δ} ½‖f + J*δ‖² subject to ‖D*δ‖ ≤ radius

where D is the scaling matrix from the cache.

# Arguments
- `strategy`: Subproblem solving strategy (QRSolve or SVDSolve)
- `J`: Jacobian matrix of the residual function
- `f`: Current residual vector
- `radius`: Trust region radius constraint
- `cache`: SubproblemCache containing factorization and scaling information

# Returns
- `λ`: Lagrange multiplier for the trust region constraint
- `δ`: Optimal step direction satisfying the trust region constraint

# Algorithm
If the Gauss-Newton step ‖δ_gn‖ ≤ radius, then λ = 0 and δ = δ_gn.
Otherwise, finds λ > 0 such that ‖D*δ‖ = radius using iterative methods.
"""
function solve_subproblem(
    strategy::SubProblemStrategy,
    J::AbstractMatrix{T},
    f::AbstractVector{T},
    radius::Real,
    cache::ColemandandLiCache,
) where {T<:Real}
    F = cache.factorization
    Dk = cache.scaling_matrix
    A = cache.Jv
    δgn = F \ -f
    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        D = √A*Dk
        λ, δ = find_λ_scaled(strategy, F, radius, J, D, f, 200, 1e-6)
    end
    return λ, δ
end
