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
        sqrt_λ = √λ
        # Augmented matrix is [J; √λ I]
        J_aug = [J; sqrt_λ * I(n)]
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


"""
TODO: read the theory and check if this is correct, for now v2 is working.
"""
function find_λ!(λ₀, Δ, J, f, maxiters)
    λ = λ₀
    p = zeros(size(J, 2))
    for i in 1:maxiters
        p = qr_regularized_solve(J, -f, λ)
        if norm(p) <= Δ
            break
        end
        #p = F \ -g
        QR = qr(J)
        q = QR.R' \ p
        #q = F'\ p
        λ = λ + (p'p)/(q'q) * (norm(p) - Δ)/Δ
    end
    println("Final step norm: ", norm(p))
    return λ
end

"""
    find_λ2!(Δ, J, f, maxiters)

This is the crucial part of the Trust Region - Levenberg-Marquardt algorithm.
When the Gauss-Newton step does not satisfy the trust region constraint,
we need to find a regularization parameter λ such that the step p satisfies:
||p|| ≈ Δ

I first saw this in Nocedal and Wright's book, Numerical Optimization, but this version is based on the 
thesis: The Levenberg-Marquardt Method and its Implementation in Python. Marius Kaltenbach, 2022.

This is an iterative method that adjusts λ so that the step p is at the radius.
Just a few iterations should be enough for the algorithm to converge (Nocedal and Wright, 2006)
 I have put 10 in the main algorithm.

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
function find_λ2!(Δ, J, f, maxiters)
    l₀ = 0.0
    u₀ = norm(J'f)/Δ
    λ₀ = maximum([1e-3*u₀,√(l₀*u₀)])
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    for i in 1:maxiters
        if (λ <= lₖ) || (λ > uₖ)
            λ = maximum([1e-3*uₖ,√(lₖ*uₖ)])
        end
        p = qr_regularized_solve(J, -f, λ)
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        end
        #p = F \ -g
        QR = qr(J, ColumnNorm())
        q = QR.P' * (QR.R' \ p)
        ϕ¹ = -(q'q)/norm(p)
        lₖ = maximum([lₖ, λ-ϕ/ϕ¹])
        #q = F'\ p
        λ = λ - (ϕ+Δ)/Δ*(ϕ/ϕ¹)
    end
    println("Final step norm: ", norm(p))
    return λ, p
end