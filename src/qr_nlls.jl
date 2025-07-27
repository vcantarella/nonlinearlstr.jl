using LinearAlgebra

"""
    QRLeastSquares

A structure to handle QR-based least squares operations efficiently.
Maintains QR factorization of the Jacobian for numerical stability.
"""
mutable struct QRLeastSquares{T}
    Q::Matrix{T}
    R::Matrix{T}
    m::Int  # number of residuals
    n::Int  # number of parameters
    rank::Int
    τ::T    # regularization parameter
end

"""
    QRLeastSquares(J::AbstractMatrix; τ::Real = 1e-12)

Initialize QR factorization from Jacobian matrix J.
"""
function QRLeastSquares(J::AbstractMatrix{T}; τ::Real = 1e-12) where T
    m, n = size(J)
    F = qr(J, ColumnNorm())
    Q = Matrix(F.Q)
    R = Matrix(F.R)
    
    # Estimate rank from diagonal of R
    rank = sum(abs.(diag(R)) .> τ * abs(R[1,1]))
    
    QRLeastSquares{T}(Q, R, m, n, rank, T(τ))
end

"""
    update!(qrls::QRLeastSquares, J::AbstractMatrix)

Update the QR factorization with a new Jacobian matrix.
"""
function update!(qrls::QRLeastSquares{T}, J::AbstractMatrix) where T
    F = qr(J)
    qrls.Q = Matrix(F.Q)
    qrls.R = Matrix(F.R)
    qrls.rank = sum(abs.(diag(qrls.R)) .> qrls.τ * abs(qrls.R[1,1]))
    return qrls
end

"""
    solve_gauss_newton(qrls::QRLeastSquares, residuals::AbstractVector)

Solve the Gauss-Newton step: J^T J δ = -J^T r using QR factorization.
This is more stable than forming J^T J explicitly.
"""
function solve_gauss_newton(qrls::QRLeastSquares{T}, residuals::AbstractVector) where T
    # For QR factorization J = QR, we have:
    # J^T J δ = -J^T r
    # R^T Q^T Q R δ = -R^T Q^T r
    # R^T R δ = -R^T Q^T r  (since Q^T Q = I)
    # R δ = -Q^T r
    
    rhs = -qrls.Q' * residuals
    
    # Handle rank deficiency
    if qrls.rank < qrls.n
        # Use pseudo-inverse for rank-deficient case
        δ = zeros(T, qrls.n)
        δ[1:qrls.rank] = qrls.R[1:qrls.rank, 1:qrls.rank] \ rhs[1:qrls.rank]
    else
        δ = qrls.R \ rhs
    end
    
    return δ
end

"""
    compute_gradient(qrls::QRLeastSquares, residuals::AbstractVector)

Compute gradient J^T r using QR factorization.
"""
function compute_gradient(qrls::QRLeastSquares, residuals::AbstractVector)
    return qrls.R' * (qrls.Q' * residuals)
end

"""
    compute_hessian_approximation(qrls::QRLeastSquares)

Compute Gauss-Newton Hessian approximation H ≈ J^T J using QR factorization.
"""
function compute_hessian_approximation(qrls::QRLeastSquares{T}) where T
    return qrls.R' * qrls.R
end

"""
    qr_trust_region_step(qrls::QRLeastSquares, residuals::AbstractVector, 
                         radius::Real, lb::AbstractVector, ub::AbstractVector,
                         x::AbstractVector; λ::Real = 0.0)

Solve trust region subproblem using QR factorization:
min_δ  0.5 * ||r + J*δ||² + 0.5*λ*||δ||²
s.t.   ||δ|| ≤ radius
       lb ≤ x + δ ≤ ub
"""
function qr_trust_region_step(qrls::QRLeastSquares{T}, residuals::AbstractVector, 
                              radius::Real, lb::AbstractVector, ub::AbstractVector,
                              x::AbstractVector; λ::Real = 0.0) where T
    
    # Augmented system for regularization:
    # [J]     [δ]   = [-r]
    # [√λ*I]         [0]
    
    if λ > 0
        # Create augmented QR system
        m, n = qrls.m, qrls.n
        J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
        r_aug = [residuals; zeros(T, n)]
        F_aug = qr(J_aug)
        Q_aug = Matrix(F_aug.Q)
        R_aug = Matrix(F_aug.R)
        
        # Solve augmented system
        rhs = -Q_aug' * r_aug
        δ_gn = R_aug \ rhs
    else
        δ_gn = solve_gauss_newton(qrls, residuals)
    end
    
    # Check bounds and trust region constraints
    δ_bounded = clamp.(δ_gn, lb - x, ub - x)
    
    if norm(δ_bounded) <= radius
        return δ_bounded
    else
        # Need to solve constrained trust region problem
        # This is a simplified version - you might want to use your tcg solver here
        δ_tr = radius * δ_gn / norm(δ_gn)
        return clamp.(δ_tr, lb - x, ub - x)
    end
end

"""
    levenberg_marquardt_step(qrls::QRLeastSquares, residuals::AbstractVector, λ::Real)

Compute Levenberg-Marquardt step: (J^T J + λI) δ = -J^T r using QR factorization.
"""
function levenberg_marquardt_step(qrls::QRLeastSquares{T}, residuals::AbstractVector, λ::Real) where T
    # Solve (J^T J + λI) δ = -J^T r
    # Using augmented system approach:
    # [J]     [δ]   = [-r]
    # [√λ*I]         [0]
    
    m, n = qrls.m, qrls.n
    J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
    r_aug = [residuals; zeros(T, n)]
    
    # QR factorization of augmented system
    F_aug = qr(J_aug)
    Q_aug = Matrix(F_aug.Q)
    R_aug = Matrix(F_aug.R)
    
    # Solve
    rhs = -Q_aug' * r_aug
    δ = R_aug \ rhs
    
    return δ
end

"""
    condition_number(qrls::QRLeastSquares)

Estimate condition number of J^T J from QR factorization.
"""
function condition_number(qrls::QRLeastSquares)
    r_diag = abs.(diag(qrls.R))
    return maximum(r_diag) / minimum(r_diag[r_diag .> qrls.τ])
end
