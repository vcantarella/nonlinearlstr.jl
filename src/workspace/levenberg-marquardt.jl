using LinearAlgebra

function solve_for_dp_dlambda(qrf, p::AbstractVector)
    # Use QR factorization to get the factors explicitly
    R = qrf.R
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    dp_dλ = UpperTriangular(R) \ (LowerTriangular(R') \ -p)
    return dp_dλ
end

function solve_for_dp_dlambda_scaled(qrf, p::AbstractVector, D::AbstractMatrix)
    # Perform the QR factorization to get the factors explicitly
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
    find_λ(Δ, J, f, maxiters; θ=1e-4)

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
function find_λ(Δ, J, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(J'f)/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    n = size(J, 2)
    p = zeros(n)
    b_aug = [-f; zeros(n)]
    for i = 1:maxiters
        qrf = qr([J; √λ * I(n)])
        p = qrf \ b_aug  # Solve the augmented system
        if (1-θ)*Δ < norm(p) < (1+θ)*Δ
            break
        end
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        @inline dpdλ = solve_for_dp_dlambda(qrf, p)
        λ = λ - (norm(p)-Δ)/Δ*(p'p)/(p'dpdλ)
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    println("Final step norm: ", norm(p))
    return λ, p
end

function find_λ_scaled(Δ, J, D, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(D*J'f)/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    b_aug = [-f; zeros(size(J, 2))]
    for i = 1:maxiters
        qrf = qr([J; √λ*D])
        p = qrf \ b_aug
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ = solve_for_dp_dlambda_scaled(qrf, p, D)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    println("Final step norm: ", norm(D*p))
    return λ, p
end
