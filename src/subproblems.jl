
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
    QRSolve <: SubProblemStrategy

Strategy that reuses QR factorization with approximation.
"""
struct QRrecursiveSolve <: SubProblemStrategy end

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
factorize(::Union{QRSolve,QRrecursiveSolve}, J) = qr(J, ColumnNorm())

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

mutable struct SubproblemCache{S<:SubProblemStrategy,F,D,T,M}
    factorization::F
    scaling_matrix::D

    # --- Buffers ---
    J_buffer::M

    p::Vector{T}             # Step direction 
    p_newton::Vector{T}      # Gradient dp/dλ

    R_buffer::Matrix{T}      # n x n mutable R
    rhs_buffer::Vector{T}    # n mutable RHS
    qtf_buffer::Vector{T}    # Stores Q'f
    v_row::Vector{T}         # Row workspace
    perm_buffer::Vector{T}   # Buffer for permutation operations

    function SubproblemCache(
        strategy::S,
        scaling_strat::Sc,
        J::AbstractMatrix{T};
        kwargs...,
    ) where {S<:SubProblemStrategy,Sc<:ScalingStrategy,T}

        m, n = size(J)
        F = factorize(strategy, J)
        Dk = scaling(scaling_strat, J; kwargs)

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

        new{S,typeof(F),typeof(Dk),T,typeof(J_buffer)}(
            F,
            Dk,
            J_buffer,
            p,
            p_newton,
            R_buffer,
            rhs_buffer,
            qtf_buffer,
            v_row,
            perm_buffer,
        )
    end
end

# --- Robust Factorize! Implementation ---

"""
    factorize!(cache, strategy, J)

Updates the factorization in `cache`. Tries to use in-place operations to reduce allocations.
"""
function factorize!(cache, strategy, J::AbstractMatrix)
    # 1. Fast Path: If we have a buffer and J is compatible, use in-place
    if cache.J_buffer !== nothing
        # Copy J into the buffer. This avoids allocating a new matrix for the input.
        copyto!(cache.J_buffer, J)

        if strategy isa Union{QRSolve,QRrecursiveSolve}
            # QR Path: In-place on J_buffer
            # Note: qr! still allocates small vectors (tau, pivots) and the wrapper struct.
            # This is O(n) alloc, saving the O(mn) matrix alloc.
            cache.factorization = qr!(cache.J_buffer, ColumnNorm())
            return cache.factorization

        elseif strategy isa SVDSolve
            # SVD Path: In-place input
            # svd! destroys the input (J_buffer) to save the copy allocation.
            # It still allocates U, S, Vt result arrays.
            cache.factorization = svd!(cache.J_buffer)
            return cache.factorization
        end
    end

    # 2. Fallback: Standard allocating version
    #    Hit if J is sparse, StaticArray, or J_buffer was not allocated.
    cache.factorization = factorize(strategy, J)
    return cache.factorization
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

    # Note: F \ -f might allocate if not careful, but usually acceptable for the check.
    # To be strictly zero-alloc, we would need to ldiv! into cache.p here.
    δgn = F \ -f
    # δgn = cache.p

    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        # PASS CACHE HERE
        λ, δ = find_λ_scaled(strategy, cache, radius, J, Dk, f, 200, 1e-6)
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
function find_λ_scaled(strategy::QRSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
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
function find_λ_scaled(strategy::SVDSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
    F = cache.factorization
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

"""
    find_λ_scaled(strategy::QRrecursiveSolve, F, Δ, J, D, f, maxiters, θ=1e-4)

Find the Lagrange multiplier λ for the scaled trust region subproblem using Recursive QR.
"""
function find_λ_scaled(strategy::QRrecursiveSolve, cache, Δ, J, D, f, maxiters, θ = 1e-4)
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
    @inbounds for i = 1:m
        qtf[i] = qtf_src[i]
    end

    # 2. Main Loop
    for i = 1:maxiters
        # FIX 3: Efficient R copy
        # Access F.factors directly if possible to avoid UpperTriangular wrapper allocs
        # If F is QRPivoted or QR, it has :factors.
        src_R = hasproperty(F, :factors) ? F.factors : F.R

        if m >= n
            @inbounds for j = 1:n, k = 1:j
                ;
                R_buffer[k, j] = src_R[k, j];
            end
        else
            fill!(R_buffer, 0.0)
            @inbounds for j = 1:n, k = 1:min(j, m)
                ;
                R_buffer[k, j] = src_R[k, j];
            end
        end

        # Reset RHS
        @inbounds for k = 1:n
            rhs_buffer[k] = -qtf[k]
        end

        # Recursive Update
        solve_damped_system_recursive_inplace!(
            p,
            R_buffer,
            rhs_buffer,
            λ,
            n,
            D,
            perm,
            v_row,
        )

        norm_Dp = norm(D*p)
        if (1-θ)*Δ < norm_Dp < (1+θ)*Δ
            break
        end

        ϕ = norm_Dp - Δ
        if ϕ < 0
            ;
            uₖ = λ;
        else
            ;
            lₖ = λ;
        end

        # Derivative
        solve_for_dp_dlambda_scaled!(dpdλ, R_buffer, p, D, perm, perm_buffer)

        # Newton Update
        denominator = (D*p)' * (D * dpdλ)
        λ = λ - (norm_Dp - Δ)/Δ * ((norm_Dp^2) / denominator)

        if !(uₖ < λ <= lₖ)
            λ = max(lₖ + 0.01*(uₖ - lₖ), √(lₖ * uₖ))
        end
    end
    return λ, p
end

function solve_damped_system_recursive_inplace!(
    p_cache,
    R_cache,
    QTr_cache,
    λ,
    n,
    D,
    perm,
    v_row,
)
    sqrt_λ = sqrt(λ)

    for c_idx = 1:n
        fill!(v_row, 0.0)

        var_idx = perm[c_idx]
        d_val = D isa Diagonal ? D[var_idx, var_idx] : D[var_idx]

        v_row[c_idx] = sqrt_λ * d_val
        v_rhs = 0.0

        for i = c_idx:n
            r_ii = R_cache[i, i]
            v_val = v_row[i]

            if abs(v_val) > 0 || i == c_idx
                c, s = compute_givens(r_ii, v_val)

                R_cache[i, i] = c * r_ii + s * v_val

                @inbounds for k = (i+1):n
                    val_R = R_cache[i, k]
                    val_v = v_row[k]
                    R_cache[i, k] = c * val_R + s * val_v
                    v_row[k] = -s * val_R + c * val_v
                end

                val_rhs_R = QTr_cache[i]
                QTr_cache[i] = c * val_rhs_R + s * v_rhs
                v_rhs = -s * val_rhs_R + c * v_rhs
            end
        end
    end

    # Use View for RHS to match dimension n
    ldiv!(p_cache, UpperTriangular(R_cache), view(QTr_cache, 1:n))

    copyto!(v_row, p_cache)
    @inbounds for i = 1:n
        p_cache[perm[i]] = v_row[i]
    end

    return p_cache
end

function solve_for_dp_dlambda_scaled!(dp_dλ, R_cache, p, D, perm, perm_buffer)
    # FIX 2: Zero-allocation D scaling
    # Replaces @. dp_dλ = -(D.diag^2) * dp_dλ
    if D isa Diagonal
        @inbounds for i = 1:length(p)
            d_val = D.diag[i]
            # Handle Bool/Number conversion implicitly by math
            val_sq = d_val * d_val
            dp_dλ[i] = -val_sq * p[i]
        end
    else
        # Fallback (allocating, but rare in this context)
        dp_dλ .= -(D' * (D * p))
    end

    # Permute RHS into perm_buffer
    @inbounds for i = 1:length(perm)
        perm_buffer[i] = dp_dλ[perm[i]]
    end

    # Solve
    ldiv!(LowerTriangular(R_cache'), perm_buffer)
    ldiv!(UpperTriangular(R_cache), perm_buffer)

    # Unpermute
    @inbounds for i = 1:length(perm)
        dp_dλ[perm[i]] = perm_buffer[i]
    end

    return dp_dλ
end

# Robust calculation of c, s for a Givens rotation
@inline function compute_givens(f::T, g::T) where {T<:Real}
    if g == 0
        return one(T), zero(T)
    end
    if f == 0
        return zero(T), one(T)
    end
    r = hypot(f, g)
    inv_r = 1 / r
    c = f * inv_r
    s = g * inv_r
    return c, s
end
