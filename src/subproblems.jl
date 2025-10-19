"""
    SubProblemStrategy

Abstract type defining strategies for solving subproblems in trust region methods.
Implementations should provide methods for factorization and solving linear systems.
"""
abstract type SubProblemStrategy end

"""
    QRSolve <: SubProblemStrategy

Strategy that uses QR factorization with column pivoting for solving subproblems.
"""
struct QRSolve <: SubProblemStrategy end

"""
    SVDSolve <: SubProblemStrategy

Strategy that uses Singular Value Decomposition (SVD) for solving subproblems.
"""
struct SVDSolve <: SubProblemStrategy end

"""
    factorize(::QRSolve, J)

Compute QR factorization with column pivoting for the Jacobian matrix J.

# Arguments
- `::QRSolve`: Strategy indicator for QR factorization
- `J`: Jacobian matrix to factorize

# Returns
- QR factorization object with column pivoting
"""
factorize(::QRSolve, J) = qr(J, ColumnNorm())

"""
    factorize(::SVDSolve, J)

Compute Singular Value Decomposition (SVD) for the Jacobian matrix J.

# Arguments
- `::SVDSolve`: Strategy indicator for SVD factorization
- `J`: Jacobian matrix to factorize

# Returns
- SVD factorization object
"""
factorize(::SVDSolve, J) = svd(J)

"""
    SubproblemCache{S<:SubProblemStrategy, F, D}

Cache structure for storing factorization and scaling information for subproblem solving.

# Type Parameters
- `S`: Subproblem strategy type (QRSolve or SVDSolve)
- `F`: Type of the factorization object
- `D`: Type of the scaling matrix

# Fields
- `factorization::F`: Cached factorization of the Jacobian matrix
- `scaling_matrix::D`: Diagonal scaling matrix for the variables

# Constructor
    SubproblemCache(strategy::S, scaling_strat::Sc, J::AbstractMatrix; kwargs...)

Create a cache object with factorization and scaling information.

# Arguments
- `strategy`: Subproblem solving strategy (QRSolve or SVDSolve)
- `scaling_strat`: Scaling strategy for the variables
- `J`: Jacobian matrix to factorize
- `kwargs...`: Additional keyword arguments passed to the scaling function
"""
mutable struct SubproblemCache{S<:SubProblemStrategy,F,D}
    factorization::F
    scaling_matrix::D
    SubproblemCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix;
        kwargs...,
    ) where {S<:SubProblemStrategy,Sc<:ScalingStrategy} = begin
        F = factorize(strategy, J)
        Dk = scaling(scaling_strat, J; kwargs)
        new{S,typeof(F),typeof(Dk)}(F, Dk)
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
    cache,
) where {T<:Real}
    F = cache.factorization
    Dk = cache.scaling_matrix
    δgn = F \ -f
    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        λ, δ = find_λ_scaled(strategy, F, radius, J, Dk, f, 200, 1e-6)
    end
    return λ, δ
end



"""
    find_λ_scaled(strategy::QRSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using QR factorization.

This function solves for λ such that ‖D*p‖ = Δ where p solves:
    (JᵀJ + λDᵀD)p = -Jᵀf

# Arguments
- `strategy::QRSolve`: QR factorization strategy
- `F`: QR factorization object (currently not used, refactored in implementation)
- `Δ`: Trust region radius
- `J`: Jacobian matrix
- `D`: Diagonal scaling matrix
- `f`: Residual vector
- `maxiters`: Maximum number of iterations for λ search
- `θ`: Tolerance for trust region constraint satisfaction (default: 1e-4)

# Returns
- `λ`: Lagrange multiplier
- `p`: Step direction satisfying ‖D*p‖ ≈ Δ

# Algorithm
Uses Newton's method with safeguarding to find λ. The search is constrained
between lower bound l₀ = 0 and upper bound u₀ = ‖D*(Jᵀf)‖/Δ.
"""
function find_λ_scaled(strategy::QRSolve, F, Δ, J, D, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    b_aug = [-f; zeros(size(J, 2))]
    for i = 1:maxiters
        F = qr([J; √λ*D])
        p = F \ b_aug
        # p = solve_augmented(strategy, J, D, b_aug, -f, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ = solve_for_dp_dlambda_scaled(strategy::QRSolve, F, p, D)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end


"""
    find_λ_scaled(strategy::SVDSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using SVD.

This function solves for λ such that ‖D*p‖ = Δ where p is computed using
the SVD factorization to solve the regularized system.

# Arguments
- `strategy::SVDSolve`: SVD factorization strategy
- `F`: SVD factorization object
- `Δ`: Trust region radius
- `J`: Jacobian matrix
- `D`: Diagonal scaling matrix
- `f`: Residual vector
- `maxiters`: Maximum number of iterations for λ search
- `θ`: Tolerance for trust region constraint satisfaction (default: 1e-4)

# Returns
- `λ`: Lagrange multiplier
- `p`: Step direction satisfying ‖D*p‖ ≈ Δ

# Algorithm
Uses Newton's method with safeguarding to find λ. The step computation
uses the SVD factorization for numerical stability with ill-conditioned systems.
"""
function find_λ_scaled(strategy::SVDSolve, F, Δ, J, D, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(D*(J'f))/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    for i = 1:maxiters
        p = solve_augmented(strategy::SVDSolve, F, J, D, -f, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ = solve_for_dp_dlambda_scaled(strategy::SVDSolve, F, D, λ, -f)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    return λ, p
end


"""
    solve_augmented(::SVDSolve, svdls::LinearAlgebra.SVD, J::AbstractMatrix, D::Diagonal, b::AbstractVector, λ::Real)

Solve the regularized linear system using SVD factorization.

Computes the solution to:
    (JᵀJ + λDᵀD)δ = b

using the SVD factorization J = UΣVᵀ. The solution is computed as:
    δ = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ

where σᵢ are singular values, dᵢ are diagonal elements of D, and uᵢ, vᵢ are
left and right singular vectors.

# Arguments
- `::SVDSolve`: SVD strategy indicator
- `svdls`: SVD factorization object containing U, Σ, V
- `J`: Jacobian matrix (not directly used, provided for interface consistency)
- `D`: Diagonal scaling matrix
- `b`: Right-hand side vector
- `λ`: Regularization parameter

# Returns
- `δ`: Solution vector to the regularized system
"""
function solve_augmented(
    ::SVDSolve,
    svdls::LinearAlgebra.SVD,
    J::AbstractMatrix,
    D::Diagonal,
    b::AbstractVector,
    λ::Real,
)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    δ = zeros(size(J, 2))
    for i = 1:n
        δ += (σs[i]/(σs[i]^2 + λ*D[i, i]^2))*(U[:, i]'*b)*V[:, i]
    end
    return δ
end

"""
    solve_for_dp_dlambda_scaled(::QRSolve, qrf::Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY}, p::AbstractVector, D::AbstractMatrix)

Compute the derivative dp/dλ for the QR-based trust region subproblem.

This function computes how the step direction p changes with respect to the
Lagrange multiplier λ. This derivative is needed for Newton's method in
finding the optimal λ.

For the system (JᵀJ + λDᵀD)p = -Jᵀf, the derivative satisfies:
    (JᵀJ + λDᵀD)(dp/dλ) = -DᵀDp

Using the QR factorization, this is solved as:
    (RᵀR)(dp/dλ) = -DᵀDp

# Arguments
- `::QRSolve`: QR strategy indicator
- `qrf`: QR factorization object
- `p`: Current step direction
- `D`: Scaling matrix

# Returns
- `dp_dλ`: Derivative of step direction with respect to λ
"""
function solve_for_dp_dlambda_scaled(
    ::QRSolve,
    qrf::Union{LinearAlgebra.QR,LinearAlgebra.QRCompactWY},
    p::AbstractVector,
    D::AbstractMatrix,
)
    # Perform the QRSolve factorization to get the factors explicitly
    R = qrf.R
    rhs = -(D'*(D*p))
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    dp_dλ = UpperTriangular(R) \ (LowerTriangular(R') \ rhs)
    return dp_dλ
end

"""
    solve_for_dp_dlambda_scaled(::SVDSolve, svdls::LinearAlgebra.SVD, D::AbstractMatrix, λ, b)

Compute the derivative dp/dλ for the SVD-based trust region subproblem.

This function computes how the step direction p changes with respect to the
Lagrange multiplier λ using the SVD factorization. The derivative is computed
analytically using the SVD representation.

For p = Σᵢ (σᵢ/(σᵢ² + λdᵢ²)) * (uᵢᵀb) * vᵢ, the derivative is:
    dp/dλ = Σᵢ (-σᵢdᵢ²/(σᵢ² + λdᵢ²)²) * (uᵢᵀb) * vᵢ

# Arguments
- `::SVDSolve`: SVD strategy indicator
- `svdls`: SVD factorization object containing U, Σ, V
- `D`: Diagonal scaling matrix
- `λ`: Current Lagrange multiplier
- `b`: Right-hand side vector

# Returns
- `dpdλ`: Derivative of step direction with respect to λ
"""
function solve_for_dp_dlambda_scaled(
    ::SVDSolve,
    svdls::LinearAlgebra.SVD,
    D::AbstractMatrix,
    λ,
    b,
)
    # Perform the SVDSolve factorization to get the factors explicitly
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    # Get diagonal elements of D
    d = diag(D)
    n = length(σs)
    # Calculate dδ/dλ using the derived formula
    dpdλ = zeros(size(V, 1))
    for i = 1:n
        coeff = -σs[i] * d[i]^2 / (σs[i]^2 + λ * d[i]^2)^2
        dpdλ += coeff * dot(U[:, i], b) * V[:, i]
    end
    return dpdλ
end
