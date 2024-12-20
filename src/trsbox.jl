"""
    trsbox(H::AbstractMatrix, g::AbstractVector, d::AbstractVector, Δ::Real, l::AbstractVector, u::AbstractVector, tol::Real, max_iter::Int)

Trust region subproblem solver based on my interpretation of the TRSBOX subproblem solver in BOBYQA. The TRSBOX subproblem solver is used in the BOBYQA optimization algorithm to solve the trust region subproblem.

# Arguments
- `H::AbstractMatrix`: The problem Hessian, or a matrix that approximates the Hessian.
- `g::AbstractVector`: The gradient of the problem.
- `Δ::Real`: The trust region radius.
- `l::AbstractVector`: The lower bound of the bounds constraints.
- `u::AbstractVector`: The upper bound of the bounds constraints.
- `tol::Real`: The tolerance for the convergence of the subproblem.
- `max_iter::Int`: The maximum number of iterations for the subproblem.

# Returns
- `d::AbstractVector`: The solution to the trust region subproblem.

# Description
We want to solve the subproblem:

    q(d) = 0.5 * d' * H * d + g' * d

where `H` is the problem Hessian, `g` is the gradient of the problem, `d` is the search direction, `Δ` is the trust region radius, `l` is the lower bound of the bounds constraints, `u` is the upper bound of the bounds constraints, `tol` is the tolerance for the convergence of the subproblem, and `max_iter` is the maximum number of iterations for the subproblem.

# Example
```julia
H = [4.0 1.0; 1.0 3.0]
g = [1.0, 2.0]
d = [0.0, 0.0]
Δ = 1.0
l = [-1.0, -1.0]
u = [1.0, 1.0]
tol = 1e-6
max_iter = 1000
d = trsbox(H, g, d, Δ, l, u, tol, max_iter)
```
"""
function trsbox(H::AbstractMatrix, grad::AbstractVector, Δ::Real, lb::AbstractVector, ub::AbstractVector, tol::Real, max_iter::Int)
    # Step 1: Initialization
    n = length(grad)
    d = zeros(n)
    g = H*d + grad
    u = -g
    s = [] # inactive set
    k = 0 #iteration counter
    f(x) = 0.5 * x' * H * x + grad' * x
    for i in 1:max_iter
        # initiating updates
        λ_Δ = Inf
        λ_lb = Inf
        λ_ub = Inf
        # Step 2: Determine the active set:
        for j in eachindex(d)
            if (abs(d[j] - lb[j])< tol) & (g[j] >= 0)
                push!(s, j) 
            elseif  (abs(d[j] - ub[j])<tol) & (g[j] <= 0)
                push!(s, j)
            end
        end
        # Step 3: Solve the trust region subproblem
        λ_cg = - (g' * u) / (u' * H * u) #step size
        update = λ_cg * u
        Pᵢ!(update, s) # make sure only active set is updated
        # Step 4. Correct solution if it is bigger than the trust region
        #    or if it is outside the bounds 
        if norm(d + update) > Δ
            λ_Δ = solve_lambdadelta(u, d, Δ)
            update = λ_Δ * u
            Pᵢ!(update, s) # only update the function at the active set:
        end
        if any(d+update .< lb)
            λ_lb = maximum((lb-d) ./ u)
            update = λ_lb * u
            Pᵢ!(update, s) # only update the function at the active set:
        end
        if any(d+update .> ub)
            λ_ub = minimum((ub-d) ./ u)
            update = λ_ub * u
            Pᵢ!(update, s) # only update the function at the active set:
        end
        # TODO: Is these next steps necessary? update again?
        # Choose the minimum as the step size
        λꜝ = minimum([λ_cg, λ_Δ, λ_lb, λ_ub])
        update = λꜝ * u
        Pᵢ!(update, s) # only update the function at the active set:

        # Preparing for the next iteration
        if norm(update) * Δ <= 0.01*(f(d)-f(d+update))
            d = d + update #update x
            println("Converged in iteration $k")
            return d
        end
        if norm(update) < tol
            d = d + update #update x
            println("Converged in iteration $k")
            return d
        end
        d = d + update #update x
        g = H*d + grad #update gradient
        β = (g' * H * u) / (u' * H * u) #update search direction
        u = -g + β * u #the search direction is a linear combination of the steepest decent and the previous search direction
        k = k + 1
    end
    println("Did not converge in $max_iter iterations")
    return d
end
"""
Pᵢ(x::AbstractVector, sᵢ::AbstractVector)
applies the activation or inactivation of dimensions based on the activation set i
"""
function Pᵢ!(x::AbstractVector, sᵢ::AbstractVector)
    for j in eachindex(sᵢ)
        x[sᵢ[j]] = 0
    end
end

"""
solve_lambdadelta(u, d, Δ, l, u)
"""
function solve_lambdadelta(u, d, Δ)
    a = u'u
    b = 2 * d'u
    c = d'd - Δ^2
    lambda = - (b + sqrt(b^2 - 4*a*c)) / (2*a)
    if lambda < 0
        lambda = - (b - sqrt(b^2 - 4*a*c)) / (2*a)
    end
    return lambda
end