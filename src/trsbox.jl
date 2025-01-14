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

    q(d) = fx + 0.5 * d' * H * d + g' * d

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
function trsbox(H::AbstractMatrix, grad::AbstractVector,
                Δ::Real, fx::Real,
                lb::AbstractVector,
                ub::AbstractVector, tol::Real,
                max_iter::Int,)
    # Step 1: Initialization
    d = zeros(eltype(grad), length(grad)) #Initial guess is zero Powell(2009)
    g = H*d + grad # gradient of the quadratic function
    u = -g
    s = Int[] # inactive set
    k = 0 #iteration counter
    f(x::AbstractVector) = fx + 0.5 * x' * H * x + grad' * x
    touch_bound = false
    for i in 1:max_iter
        # Step 1: Update active set
        # empty!(s)  # Clear previous active set
        for j in eachindex(d)
            if (abs(d[j] - lb[j]) < tol && g[j] >= 0) || 
               (abs(d[j] - ub[j]) < tol && g[j] <= 0)
                push!(s, j)
            end
        end
        Pᵢ!(u, s) # make sure only active set is updated

        # Step 3: Compute the CG step (Solve the trust region subproblem)
        Hu = H * u
        u_Hu = u' * Hu

        g_u = g' * u
        λ_cg = -g_u / u_Hu
        λ_Δ = solve_lambdadelta(u, d, Δ)
        λ_lb = maximum((lb-d) ./ u)
        λ_ub = maximum((ub-d) ./ u)
        λ = minimum([λ_cg, λ_Δ, λ_lb, λ_ub])
        index = argmin([λ_cg, λ_Δ, λ_lb, λ_ub])
        d = d + λ * u
        if index == 2
            println("Converged at the trust region boundary")
            return d
        elseif index == 3 || index == 4
            # Pᵢ!(d, s)
            red = f(zeros(size(d))) - f(d)
            if f(d)*Δ < 0.01*red
                return d
            else
                g = H*d + grad
                u = -g
                k = k + 1
                continue
            end
        else # index == 1
            g = H*d + grad
            if norm(d)^2 * norm(g)^2 - (d'g)^2 <= 1e-4*(f(zeros(size(d)))-f(d))^2
                return d
            end
            β = (g' * Hu) / (u_Hu) #update search direction
            u = -g + β * u #the search direction is a linear combination of the steepest decent and the previous search direction
            k = k + 1
        end

        # # Step 4. Correct solution if it is bigger than the trust region
        # #    or if it is outside the bounds 
        # if abs(norm(d + λ*u) - Δ) > tol
        #     # if norm(d + u) > Δ
        #     #     u = -g / norm(g)
        #     # end
        #     if norm(d) > Δ
        #         error("The solution is outside the trust region something went wrong")
        #     end
        #     λ_Δ = solve_lambdadelta(u, d, Δ)
        #     update = λ_Δ * u
        #     println("Converged at the radius boundary")
        #     return d + update
        # end

        # # Preparing for the next iteration
        # if touch_bound & (norm(update) * Δ <= 0.01*(f(d)-f(d+update)))
        #     d = d + update #update x
        #     println("Converged in iteration $k")
        #     return d
        # end

        # update = λ * u
        # d = d + update #update x
        # g = H*d + grad #update gradient

        # # Verify constraints are satisfied
        # @assert all(lb .<= d_new .<= ub) "Bound constraints violated"
        # @assert norm(d_new) <= Δ + tol "Trust region constraint violated"
        # if norm(g) < tol
        #     println("Converged in iteration $k")
        #     return d
        # end
        # β = (g' * Hu) / (u_Hu) #update search direction
        # u = -g + β * u #the search direction is a linear combination of the steepest decent and the previous search direction
        # k = k + 1
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