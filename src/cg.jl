"""
    cg(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector, tol::Real, max_iter::Int)

Solve the linear system `Ax = b` using the Conjugate Gradient method.

# Arguments
- `A::AbstractMatrix`: A symmetric positive definite matrix.
- `b::AbstractVector`: A vector.
- `x0::AbstractVector`: Initial guess for the solution.
- `tol::Real`: Tolerance for the stopping criterion.
- `max_iter::Int`: Maximum number of iterations.

# Returns
- `x::AbstractVector`: The approximate solution to the system `Ax = b`.

# Description
The Conjugate Gradient method is an iterative algorithm for solving systems of linear equations with a symmetric positive definite matrix. The quadratic form minimized is:

    q(x) = 0.5 * x' * A * x + b' * x + c

where `A` is a symmetric positive definite matrix, `b` is a vector, and `c` is a scalar.

# Example
```julia
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
x0 = [2.0, 1.0]
tol = 1e-6
max_iter = 1000
x = cg(A, b, x0, tol, max_iter)
```
"""
function cg(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector, tol::Real, max_iter::Int)
    x = x0
    g = A*x + b #g_0 initial gradient
    u = -g #u_0 initial search direction is the steepest decent
    k = 0 #iteration counter
    for i in 1:max_iter
        λꜝ = - (g' * u) / (u' * A * u) #step size
        x = x + λꜝ * u #update x
        g = A*x + b #update gradient
        if norm(g) < tol
            return x
        end
        β = (g' * A * u) / (u' * A * u) #update search direction
        u = -g + β * u #the search direction is a linear combination of the steepest decent and the previous search direction
        k = k + 1
    end
    return x
end
