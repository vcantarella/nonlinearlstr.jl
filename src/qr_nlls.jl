using LinearAlgebra

"""
    solve_gauss_newton(qrls::QRLeastSquares, residuals::AbstractVector)

Solve the Gauss-Newton step: J^T J δ = -J^T r using QR factorization.
This is more stable than forming J^T J explicitly.
"""
function solve_gauss_newton(J, qrls::QRPivoted, residuals::AbstractVector, 
                            τ::Real = 1e-12)
    # For QR factorization J = QR, we have:
    # J^T J δ = -J^T r
    # R^T Q^T Q R δ = -R^T Q^T r  
    # R^T R δ = -R^T Q^T r  (since Q^T Q = I)
    # R δ = -Q^T r

    # m, n = size(J)  # J is m×n (m residuals, n parameters)
    # rhs = - vec(qrls.Q' * residuals)  # Q' is n×m, residuals is m×1, so rhs is n×1

    # # Calculate rank using maximum diagonal element
    # r_diag = abs.(diag(qrls.R))
    # max_diag = maximum(r_diag)
    # rank = sum(r_diag .> τ * max_diag)
    
    # # Handle rank deficiency
    # if rank < n
    #     # Use pseudo-inverse for rank-deficient case
    #     δ_permuted = zeros(n)
    #     δ_permuted[1:rank] = UpperTriangular(qrls.R[1:rank, 1:rank]) \ rhs[1:rank]
        
    #     # Apply inverse permutation: if p[i] = j, then unpermuted[j] = permuted[i]
    #     δ = zeros(n)
    #     δ[qrls.p] = δ_permuted
    # else
        # Full rank case
        # δ_permuted = UpperTriangular(qrls.R) \ rhs
        
        # # Apply inverse permutation
        # δ = zeros(n)
        # δ[qrls.p] = δ_permuted
        δ = J \ (-residuals)  # Solve J δ = -r directly for full rank
    # end

    return δ
end

"""
    compute_gradient(qrls::QRLeastSquares, residuals::AbstractVector)

Compute gradient J^T r using QR factorization.
"""
function compute_gradient(J, residuals::AbstractVector)
    return J' * residuals
end

"""
    compute_hessian_approximation(qrls::QRLeastSquares)

Compute Gauss-Newton Hessian approximation H ≈ J^T J using QR factorization.
"""
function compute_hessian_approximation(qrls::QRPivoted)
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
function qr_trust_region_step(J, qrls::QRPivoted, residuals::AbstractVector, 
                              radius::Real, lb::AbstractVector, ub::AbstractVector,
                              x::AbstractVector; λ::Real = 0.0)
    # Augmented system for regularization:
    # [J]     [δ]   = [-r]
    # [√λ*I]         [0]
    
    if λ > 0
        # Create augmented QR system
        m, n = size(J)
        J_aug = [qrls.Q * qrls.R; sqrt(λ) * I(n)]
        r_aug = [residuals; zeros(n)]
        # F_aug = qr(J_aug, ColumnNorm())
        # Q_aug = F_aug.Q
        # R_aug = UpperTriangular(F_aug.R)

        # # Solve augmented system
        # rhs = - (Q_aug' * r_aug)
        δ_gn = J_aug \ (-r_aug)
        #δ_gn = δ_gn[qrls.p]  # Apply permutation
    else
        δ_gn = solve_gauss_newton(J, qrls, residuals)
    end
    
    # Check bounds and trust region constraints
    δ_bounded = clamp.(δ_gn, lb - x, ub - x)
    # if norm(δ_bounded) <= radius
    #     return δ_bounded
    # else
    #     # Need to solve constrained trust region problem
    #     # This is a simplified version - you might want to use your tcg solver here
    #     δ_tr = radius * δ_gn / norm(δ_gn)
    #     return clamp.(δ_tr, lb - x, ub - x)
    # end
    return δ_bounded
end

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


function levenberg(
    res::Function, jac::Function,
    x0::Array{T},
    lb::Array{T}, ub::Array{T};
    max_trust_radius = nothing,
    min_trust_radius = 1e-12,
    initial_radius = 1.0,
    step_threshold = 0.01,
    shrink_threshold = 0.25,
    expand_threshold = 0.9,
    shrink_factor = 0.25,
    expand_factor = 2.0,
    Beta = 0.99999,
    max_iter = 100,
    gtol = 1e-6,
    ) where T
    # Check if x0 is within bounds
    if any(x0 .< lb) || any(x0 .> ub)
        error("Initial guess x0 is not within bounds")
    end

    r = res(x0)
    f0 = 1/2 * r' * r
    if max_trust_radius === nothing
        max_radius = max(norm(f0), maximum(x0) - minimum(x0))
    else
        max_radius = max_trust_radius
    end

    J = jac(x0)
    qrls = qr(J, ColumnNorm())

    g0 = compute_gradient(qrls, r)

    Bk = compute_hessian_approximation(qrls)

    #Step 1 termination test:
    if (norm(g0) < gtol) || (initial_radius < min_trust_radius)
        println("Initial guess is already a minimum gradientwise")
        return x0, f0, g0, 0
    end

    ## initializing the scaling variables:
    ak = zeros(eltype(x0), length(x0))
    bk = zeros(eltype(x0), length(x0))
    Dk = Diagonal(ones(eltype(x0), length(x0)))
    inv_Dk = copy(Dk)
    affine_cache = (ak = ak, bk = bk, Dk = Dk, inv_Dk = inv_Dk)

    # initializing the x and g vectors
    x = x0
    g = g0
    f = f0
    radius = initial_radius

    for iter in 1:max_iter
        #Step 2: Determine trial step
        ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
        ## Step 2.1 Calculate the scaling vectors and the scaling matrix
        update_Dk!(affine_cache, x, lb, ub, g, radius, 1e-16)
        Dk = affine_cache.Dk
        inv_Dk = affine_cache.inv_Dk
        ## Step 2.2: Solve the trust region subproblem in the scaled space: d_hat = inv(D)*d
        # f_dhat(d_hat) = (Dk*g)'*d_hat + 0.5*d_hat'*(Dk*Bk*Dk)*d_hat
        dhatl = inv_Dk * (lb - x)
        dhatu = inv_Dk * (ub - x)
        J = J*Dk
        F = qr(J, ColumnNorm())
        A = Dk*Bk*Dk
        b = Dk*g
        d_hat = zeros(eltype(x0), length(x0))
        try
            d_hat = tcg(A, b, radius, dhatl, dhatu, gtol, 1000)
        catch e
            if isa(e, DomainError)
                println("Domain error encountered: ", e)
                radius = 0.5 * radius
                if (radius < min_trust_radius) || (norm(g) < gtol)
                    # print gradient convergence
                    println("Gradient convergence criterion reached")
                    return x, f, g, iter
                end
                continue
            else
                rethrow(e)
            end
        end
        sk = Beta .* Dk * d_hat

        if norm(sk) < gtol
            println("Step size convergence criterion reached")
            return x, f, g, iter
        end

        f_new = func(x+sk)
        actual_reduction = f - f_new
        if actual_reduction < 0
            radius = 0.5 * radius
            if (radius < min_trust_radius) || (norm(g) < gtol)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            continue
        end
        pred_reduction = -(0.5 * sk' * Bk * sk + g' * sk)
        if pred_reduction < 0
            radius = 0.5 * radius
            if (radius < min_trust_radius) || (norm(g) < gtol)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            continue
        end
        ρ = actual_reduction / pred_reduction

        if ρ ≥ step_threshold
            x = x + sk
            f = f_new
            g = grad(x)
            Bk = hess(x)
            println("Iteration: $iter, f: $f, norm(g): $(norm(g))")
            println("--------------------------------------------")
        #Step 4: Update the trust region radius
            if ρ > expand_threshold
                radius = maximum([radius, expand_factor * norm(inv_Dk*sk)])
                if radius > max_radius
                    radius = max_radius
                end
            elseif ρ < shrink_threshold
                maximum([0.5*radius, shrink_factor * norm(inv_Dk*sk)])
            end
        else
            radius = 0.5 * radius
        end
        if (radius < min_trust_radius) || (norm(g) < gtol)
            # print gradient convergence
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end