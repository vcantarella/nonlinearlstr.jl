using LinearAlgebra

"""
    svd_regularized_solve(svdls, b, λ::Real)

Solve a regularized linear least squares problem using SVD decomposition.
    solution in for example: https://en.wikipedia.org/wiki/Ridge_regression
This function solves the regularized system:
    min ||Jx - b||² + λ||x||²

using the SVD decomposition and solving the augmented system:
# Arguments
- `svdls`: The SVD factorization of the Jacobian matrixs
- `b::AbstractVector`: The right-hand side vector (length m)
- `λ::Real`: Regularization parameter. If λ > 0, Tikhonov regularization is applied

# Returns
- `x::Vector`: Solution vector of length n
"""
function svd_regularized_solve(svdls, b, λ)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    x_svd = sum([σs[i]/(σs[i]^2 + λ)*U[:, i]'*b*V[:, i] for i = 1:n])
    return x_svd
end


"""
    svd_regularized_solve_scaled(svdls, b, λ::Real, d)

Solve a regularized linear least squares problem using SVD decomposition.
    solution in for example: https://en.wikipedia.org/wiki/Ridge_regression
This function solves the regularized system
at the scaling D = diag(d):
    min ||Jx - b||² + λ||Dx||²
using the SVD decomposition and solving the augmented system:
# Arguments
- `svdls`: The SVD factorization of the Jacobian matrixs
- `b::AbstractVector`: The right-hand side vector (length m)
- `λ::Real`: Regularization parameter. If λ > 0, Tikhonov regularization is applied
- `d::AbstractVector`: Scaling vector with diagonal entries of the scaling matrix D (length n)
# Returns
- `x::Vector`: Solution vector of length n
"""
function svd_regularized_solve_scaled!(svdls, b, λ, d)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    x_svd = sum([σs[i]/(σs[i]^2 + λ*d[i]^2)*U[:, i]'*b*V[:, i] for i = 1:n])
    return x_svd
end

"""
    solve_for_dp_dlambda(svdls, b, λ::Real)
this function computes the derivative of the solution p
     with respect to the regularization parameter λ.
It is a simple handwritten derivative from the above expression
 for p (in svd_regularized_solve).
"""
function solve_for_dpdλ(svdls, b, λ)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    dpdλ = sum([-σs[i]/(σs[i]^2 + λ)^2*U[:, i]'*b*V[:, i] for i = 1:n])
    return dpdλ
end

#TODO: implement this for svd factorization
function solve_for_dp_dlambda_scaled(svdls, b, λ)
    #dpdλ
    return nothing
end


"""
    find_λ!(Δ, svdls, J, f, maxiters; θ=1e-4)

This is the crucial part of the Trust Region - Levenberg-Marquardt algorithm.
When the Gauss-Newton step does not satisfy the trust region constraint,
we need to find a regularization parameter λ such that the step p satisfies:
||p(λ)|| ≈ Δ

This is based in Algorithm 4.3 Nocedal and Wright's book, Numerical Optimization.
This version borrow some ideas from the thesis: The Levenberg-Marquardt Method
    and its Implementation in Python. Marius Kaltenbach, 2022.
The main difference is that this algorithm uses the SVD factorization to find p 
and the derivative dpdλ.
This allows us to reuse the initial factorization of J,
 therefore the whole algorithm just factorizes once per iteration!
Which is a major advantage over the QR decomposition, which needs to refactorize at
    each λ iteration.
With the SVD factorization we dont need to invert matrices explicitly, which saves computation time.

The method implements the idea of solving the Newton method for λ using a different approach.
This is an iterative method that adjusts λ so that the step p is at the radius.
Just a few iterations should be enough for the algorithm to converge (Nocedal and Wright, 2006).
First it finds the steps for a given λ,
then it updates with a newton iteration, λₖ₊₁ = λₖ - (ϕ₂/ϕ₂'),
where ϕ₂ = 1/Δ-1/||p||: secant formula,

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
function find_λ_svd(Δ, svdls, J, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(J'f)/Δ
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    dpdλ = zeros(size(J, 2))
    for i = 1:maxiters
        p .= svd_regularized_solve(svdls, -f, λ)
        if (1-θ)*Δ < norm(p) < (1+θ)*Δ
            break
        end
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ .= solve_for_dpdλ(svdls, -f, λ)
        λ = λ - (norm(p)-Δ)/Δ*(p'p)/(p'dpdλ)
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    println("Final step norm: ", norm(p))
    return λ, p
end
