using LinearAlgebra

"""
    solve_gauss_newton(J, r)

Solve the Gauss-Newton step: J^T J δ = -J^T r using QR factorization.
This is more stable than forming J^T J explicitly.
"""
function solve_gauss_newton(J, r::AbstractVector)
    δ = J \ (-r)  # Solve J δ = -r directly for full rank (implemented as QR by default)
    return δ
end

"""
    compute_gradient(J, r::AbstractVector)

Compute gradient J^T r of the loss function L(x) = 1/2 ||r||^2.
"""
function compute_gradient(J, r::AbstractVector)
    return J' * r
end

"""
    compute_hessian_approximation(qrls::QRPivoted)

Compute Gauss-Newton Hessian approximation H ≈ J^T J using QR factorization.
TODO: Check if we need to correct for pivoting in the QR factorization.
"""
function compute_hessian_approximation(qrls::QRPivoted)
    return qrls.R' * qrls.R
end

# """
#     qr_trust_region_step(qrls::QRLeastSquares, residuals::AbstractVector, 
#                          radius::Real, lb::AbstractVector, ub::AbstractVector,
#                          x::AbstractVector; λ::Real = 0.0)

# Solve trust region subproblem using QR factorization:
# min_δ  0.5 * ||r + J*δ||² + 0.5*λ*||δ||²
# s.t.   ||δ|| ≤ radius
#        lb ≤ x + δ ≤ ub
# """
# function qr_trust_region_step(J, qrls::QRPivoted, residuals::AbstractVector, 
#                               radius::Real, lb::AbstractVector, ub::AbstractVector,
#                               x::AbstractVector; λ::Real = 0.0)
#     # Augmented system for regularization:
#     # [J]     [δ]   = [-r]
#     # [√λ*I]         [0]
    
#     if λ > 0
#         # Create augmented QR system
#         m, n = size(J)
#         J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
#         r_aug = [residuals; zeros(n)]
#         # F_aug = qr(J_aug, ColumnNorm())
#         # Q_aug = F_aug.Q
#         # R_aug = UpperTriangular(F_aug.R)

#         # # Solve augmented system
#         # rhs = - (Q_aug' * r_aug)
#         δ_gn = J_aug \ (-r_aug)
#         #δ_gn = δ_gn[qrls.p]  # Apply permutation
#     else
#         δ_gn = solve_gauss_newton(J, qrls, residuals)
#     end
    
#     # Check bounds and trust region constraints
#     δ_bounded = clamp.(δ_gn, lb - x, ub - x)
#     # if norm(δ_bounded) <= radius
#     #     return δ_bounded
#     # else
#     #     # Need to solve constrained trust region problem
#     #     # This is a simplified version - you might want to use your tcg solver here
#     #     δ_tr = radius * δ_gn / norm(δ_gn)
#     #     return clamp.(δ_tr, lb - x, ub - x)
#     # end
#     return δ_bounded
# end

# """
#     levenberg_marquardt_step(qrls::QRLeastSquares, residuals::AbstractVector, λ::Real)

# Compute Levenberg-Marquardt step: (J^T J + λI) δ = -J^T r using QR factorization.
# """
# function levenberg_marquardt_step(qrls::QRLeastSquares{T}, residuals::AbstractVector, λ::Real) where T
#     # Solve (J^T J + λI) δ = -J^T r
#     # Using augmented system approach:
#     # [J]     [δ]   = [-r]
#     # [√λ*I]         [0]
    
#     m, n = qrls.m, qrls.n
#     J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
#     r_aug = [residuals; zeros(T, n)]
    
#     # QR factorization of augmented system
#     F_aug = qr(J_aug)
#     Q_aug = Matrix(F_aug.Q)
#     R_aug = Matrix(F_aug.R)
    
#     # Solve
#     rhs = -Q_aug' * r_aug
#     δ = R_aug \ rhs
    
#     return δ
# end

"""
    condition_number(qrls::QRLeastSquares)

Estimate condition number of J^T J from QR factorization.
"""
function condition_number(qrls::QRPivoted, τ::Real = 1e-12)
    r_diag = abs.(diag(qrls.R))
    return maximum(r_diag) / minimum(r_diag[r_diag .> τ])
end