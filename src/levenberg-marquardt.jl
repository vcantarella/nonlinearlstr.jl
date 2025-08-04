using LinearAlgebra

"""
    qr_regularized_solve(J::AbstractMatrix, b::AbstractVector, λ::Real)

Solve a regularized linear least squares problem using QR decomposition.

This function solves the regularized system:
    min ||Jx - b||² + λ||x||²

by forming and solving the augmented system:
    [J; √λ I][x] = [b; 0]

# Arguments
- `J::AbstractMatrix`: The Jacobian matrix (m×n)
- `b::AbstractVector`: The right-hand side vector (length m)
- `λ::Real`: Regularization parameter. If λ > 0, Tikhonov regularization is applied

# Returns
- `x::Vector`: Solution vector of length n

# Details
When λ > 0, the function creates an augmented system by appending √λ times the 
identity matrix to J and padding b with zeros. This transforms the regularized 
problem into a standard least squares problem that can be solved directly.

When λ = 0, the function reduces to solving the unregularized system Jx = b.

# Examples
"""
function qr_regularized_solve(J::AbstractMatrix, b::AbstractVector, λ::Real)
    m, n = size(J)
    
    # Form augmented system: [J; √λ I][x] = [b; 0]
    if λ > 0
        # Augmented matrix is [J; √λ I]
        J_aug = [J; √λ * I(n)]
        b_aug = [b; zeros(n)]
    else
        J_aug = J
        b_aug = b
    end
    
    # QR factorization of augmented system
    # Solve the system
    x = J_aug \ b_aug # LinearAlgebra's backslash operator uses QR by default for full rank matrices
    return x
end

# A new function to solve for dp/dλ using the same QR factorization approach
# Note: It is more efficient to pass the QR factorization object itself,
# but for clarity, we can re-factorize inside.
"""
    solve_for_dp_dlambda(J::AbstractMatrix, p::AbstractVector, λ::Real)
this function computes the derivative of the solution p with respect to the regularization parameter λ.

"""
function solve_for_dp_dlambda(J::AbstractMatrix, p::AbstractVector, λ::Real)
    n = size(J,2)
    # Form the augmented matrix, just like before
    J_aug = [J; √λ * I(n)]
    
    # Perform the QR factorization to get the factors explicitly
    Q, R = qr(J_aug)
    
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    
    # In Julia, this is easily expressed as:
    dp_dλ = UpperTriangular(R) \ (LowerTriangular(R') \ -p)
    return dp_dλ
end

function qr_regularized_solve2(J::AbstractMatrix, b::AbstractVector, λ::Real, D)
    m, n = size(J)
    
    # Form augmented system: [J; √λ I][x] = [b; 0]
    if λ > 0
        # Augmented matrix is [J; √λ I]
        J_aug = [J; √λ * D]
        b_aug = [b; zeros(n)]
    else
        J_aug = J
        b_aug = b
    end
    
    # QR factorization of augmented system
    # Solve the system
    x = J_aug \ b_aug # LinearAlgebra's backslash operator uses QR by default for full rank matrices
    return x
end


function solve_for_dp_dlambda2(J::AbstractMatrix, p::AbstractVector, λ::Real, D)
    n = size(J,2)
    # Form the augmented matrix, just like before
    J_aug = [J; √λ * D]
    
    # Perform the QR factorization to get the factors explicitly
    Q, R = qr(J_aug)
    
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    
    # In Julia, this is easily expressed as:
    dp_dλ = UpperTriangular(R) \ (LowerTriangular(R') \ (-D^2*p))
    return dp_dλ
end

"""
    find_λ!(Δ, J, f, maxiters; θ=1e-4)

This is the crucial part of the Trust Region - Levenberg-Marquardt algorithm.
When the Gauss-Newton step does not satisfy the trust region constraint,
we need to find a regularization parameter λ such that the step p satisfies:
||p(λ)|| ≈ Δ

This is based in Algorithm 4.3 Nocedal and Wright's book, Numerical Optimization.
This version borrow some ideas from the thesis: The Levenberg-Marquardt Method
    and its Implementation in Python. Marius Kaltenbach, 2022.

However, it implements the idea of solving the Newton method for λ using a different approach, by
solving with the QR factorization of the augmented system, instead of the Cholesky factorization.
This is more stable than forming J'J explicitly.

This is an iterative method that adjusts λ so that the step p is at the radius.
Just a few iterations should be enough for the algorithm to converge (Nocedal and Wright, 2006).

First it finds the steps for a given λ,
then it updates with a newton iteration, λₖ₊₁ = λₖ - (ϕ₂/ϕ₂'),
where ϕ₂ = 1/Δ-1/||p||,

# Arguments
- `Δ::Real`: Trust region radius (constraint on step norm)
- `J::AbstractMatrix`: Jacobian matrix 
- `f::AbstractVector`: Function values/residuals
- `maxiters::Integer`: Maximum number of iterations

# Returns
- `λ::Real`: The computed Levenberg-Marquardt parameter
- `p::AbstractVector`: The computed step vector satisfying the trust region constraint

# Algorithm
The method maintains lower (`lₖ`) and upper (`uₖ`) bounds on λ and uses:
1. QR factorization with regularization to solve the linear system
2. Newton-like updates for λ based on the constraint violation ϕ = ||p|| - Δ
3. Safeguarding to ensure λ remains within reasonable bounds

# Notes
- The function modifies internal variables during iteration
- Prints the final step norm for debugging purposes
- Uses `qr_regularized_solve` for the regularized linear solve

# References
- Nocedal, J., & Wright, S. (2006). Numerical Optimization.
- KALTENBACH, Marius, 2022. The Levenberg-Marquardt Method and 
    its Implementation in Python [Master thesis]. Konstanz: Universität Konstanz
"""
function find_λ!(Δ, J, f, maxiters, θ=1e-4)
    l₀ = 0.0
    u₀ = norm(J'f)/Δ
    λ₀ = maximum([1e-3*u₀,√(l₀*u₀)])
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    for i in 1:maxiters
        if !(uₖ < λ <= lₖ)
            λ = maximum([1e-3*uₖ,√(lₖ*uₖ)])
        end
        p = qr_regularized_solve(J, -f, λ)
        if (1-θ)*Δ < norm(p) < (1+θ)*Δ
            break
        end
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        end
        dpdλ = solve_for_dp_dlambda(J, p, λ)
        λ = λ - (norm(p)-Δ)/Δ*(p'p)/(p'dpdλ)
        lₖ = maximum([lₖ, λ])
    end
    println("Final step norm: ", norm(p))
    return λ, p
end

# function find_λ2!(Δ, J, f, maxiters, θ=1e-4, D, D_inv)
#     l₀ = 0.0
#     u₀ = norm(D_inv*J'*f)/Δ
#     λ₀ = maximum([1e-3*u₀,√(l₀*u₀)])
#     λ = λ₀
#     uₖ = u₀
#     lₖ = l₀
#     p = zeros(size(J, 2))
#     for i in 1:maxiters
#         if !(uₖ < λ <= lₖ)
#             λ = maximum([1e-3*uₖ,√(lₖ*uₖ)])
#         end
#         p = qr_regularized_solve2(J, -f, λ, D)
#         if (1-θ)*Δ < norm(p) < (1+θ)*Δ
#             break
#         end
#         ϕ = norm(p)-Δ
#         if ϕ < 0
#             uₖ = λ
#         end
#         dpdλ = solve_for_dp_dlambda2(J, p, λ, D)
#         λ = λ - (norm(p)-Δ)/Δ*(p'p)/(p'dpdλ)
#         lₖ = maximum([lₖ, λ])
#     end
#     println("Final step norm: ", norm(p))
#     return λ, p
# end

# function find_λ3!(Δ, J, D, f, maxiters, θ=1e-4)
#     l₀ = 0.0
#     u₀ = norm(J'f)/Δ
#     λ₀ = maximum([1e-3*u₀,√(l₀*u₀)])
#     λ = λ₀
#     uₖ = u₀
#     lₖ = l₀
#     p = zeros(size(J, 2))
#     for i in 1:maxiters
#         if (λ <= lₖ) || (λ > uₖ)
#             λ = maximum([1e-3*uₖ,√(lₖ*uₖ)])
#         end
#         p = qr_regularized_solve2(J, -f, λ, D)
#         ϕ = norm(D*p)-Δ
#         if ϕ < 0
#             uₖ = λ
#         end
#         #p = F \ -g
#         QR = qr(J, ColumnNorm())
#         q = QR.P' * (QR.R' \ (D*D*p))
#         ϕ¹ = -(q'q)/norm(D*p)
#         lₖ = maximum([lₖ, λ-ϕ/ϕ¹])
#         #q = F'\ p
#         λ = λ - (ϕ+Δ)/Δ*(ϕ/ϕ¹)
#         if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
#             break
#         end
#     end
#     println("Final step norm: ", norm(D*p))
#     return λ, p
# end