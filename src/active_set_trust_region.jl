using LinearAlgebra

"""
    solve_substep(J, f, radius) -> (λ, δ)

Solve the trust region subproblem of minimizing ||J δ + f|| subject to ||δ|| ≤ radius.

This function uses the Singular Value Decomposition (SVD) of the Jacobian matrix J to 
efficiently solve the constrained optimization problem that arises in trust region methods.

# Arguments
- `J`: Jacobian matrix
- `f`: Function residual vector
- `radius`: Trust region radius constraint

# Returns
- `λ`: Damping parameter (Lagrange multiplier). If λ = 0, the step perfectly fits the model
- `δ`: Computed step vector satisfying the trust region constraint

# Algorithm
1. Computes the Gauss-Newton step δgn = -J⁻¹f using SVD
2. If ||δgn|| ≤ radius, returns the unconstrained solution with λ = 0
3. Otherwise, finds a damped step on the trust region boundary using Levenberg-Marquardt approach

# Notes
- When λ = 0, the returned step δ minimizes the model exactly within the trust region
- When λ > 0, the step lies on the trust region boundary and represents a compromise 
  between model reduction and step size constraint
"""
function solve_substep(J, f, radius)
    svdls = svd(J)
    δgn = svdls \ -f
        if norm(δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ_svd(radius, svdls, J, f, 100)
        end
    return λ, δ
end



"""
    solve_bounded_subproblemm(J, f, radius, lb, ub, maxiters=20)

Solve a bounded trust region subproblem using an active set method.

This function solves the constrained optimization subproblem:
    minimize ½‖f + Jδ‖² subject to ‖δ‖ ≤ radius and lb ≤ δ ≤ ub

The algorithm uses an iterative active set approach:
1. Computes an initial trial solution using the unconstrained trust region solver
2. Identifies the active and inactive constraint sets based on the gradient
3. Iteratively refines the solution by solving reduced subproblems on the inactive set
4. Updates the active set until convergence or maximum iterations reached

# Arguments
- `J`: Jacobian matrix
- `f`: Residual vector
- `radius`: Trust region radius constraint
- `lb`: Lower bounds vector for the variables
- `ub`: Upper bounds vector for the variables  
- `maxiters=20`: Maximum number of active set iterations

# Returns
- `δ`: Solution vector satisfying the bound and trust region constraints

# Notes
- Convergence is achieved when either the active set stabilizes or the solution changes negligibly
- A warning is printed if maximum iterations are reached without convergence
- The function assumes `solve_substep(J, f, radius)` is available for solving unconstrained trust region subproblems
"""
function solve_bounded_subproblemm(J, f, radius, lb, ub, maxiters=20)
    
    #trial solution
    λ, δ = solve_substep(J, f, radius)
    # evaluate the inactive set
    Jδ = J * δ
    g = J'*(Jδ + f) # gradient ∇ₓ of ½‖f + Jδ‖²
    inactive = Bool[lb[i] < ub[i] && (δ[i] != lb[i] || g[i] ≤ 0) &&
                    (δ[i] != ub[i] || g[i] ≥ 0) for i in eachindex(δ)]
    all(inactive) && return λ, δ
    active = map(!, inactive)
    δprev = copy(δ)
    for iter = 1:maxiters
        λa, δa = solve_substep(J[:,inactive], f + J[:,active]*δ[active], radius)
        δ[inactive] = δa
        @. δ = clamp(δ, lb, ub)
        λ += λa # update the Lagrange multipliers (not sure if this is correct)
        g .= mul!(g, J', J*δ) .+ J'f
        for i in eachindex(δ)
            inactive[i] = lb[i] < ub[i] && (δ[i] != lb[i] || g[i] ≤ 0) &&
                (δ[i] != ub[i] || g[i] ≥ 0)
        end
        all(i -> inactive[i] == !active[i], eachindex(active)) && return λ, δ # convergence: active set unchanged 
        norm(δ - δprev) ≤ eps(float(eltype(δ)))*sqrt(length(δ)) && return λ, δ # convergence: x not changing much
        δprev .= δ
        @. active = !inactive
    end
    println("Warning!: convergence failure: $maxiters iterations reached")
    return λ, δ
end

"""
    active_set_svd_trust_region(res, jac, x0; kwargs...)

Solve a nonlinear least squares problem using an active set trust region method with SVD-based subproblem solving.

This function minimizes the objective function `0.5 * ||res(x)||²` subject to box constraints `lb ≤ x ≤ ub` 
using a trust region approach combined with an active set strategy for handling bounds.

# Arguments
- `res::Function`: Residual function that takes a vector `x` and returns the residual vector
- `jac::Function`: Jacobian function that takes a vector `x` and returns the Jacobian matrix
- `x0::Array{T}`: Initial guess for the solution

# Keyword Arguments
- `lb::Array{T}`: Lower bounds for variables (default: `-Inf` for all variables)
- `ub::Array{T}`: Upper bounds for variables (default: `+Inf` for all variables)
- `initial_radius::Real`: Initial trust region radius (default: `1.0`, auto-adjusted based on `||x0||`)
- `max_trust_radius::Real`: Maximum allowed trust region radius (default: `1e12`)
- `min_trust_radius::Real`: Minimum trust region radius before termination (default: `1e-8`)
- `step_threshold::Real`: Minimum reduction ratio to accept a step (default: `0.01`)
- `shrink_threshold::Real`: Reduction ratio threshold for shrinking trust region (default: `0.25`)
- `expand_threshold::Real`: Reduction ratio threshold for expanding trust region (default: `0.75`)
- `shrink_factor::Real`: Factor to shrink trust region radius (default: `0.25`)
- `expand_factor::Real`: Factor to expand trust region radius (default: `2.0`)
- `max_iter::Int`: Maximum number of iterations (default: `100`)
- `gtol::Real`: Gradient tolerance for convergence (default: `1e-6`)
- `ftol::Real`: Function tolerance for convergence (default: `1e-15`)
- `τ::Real`: Regularization parameter (default: `1e-12`)

# Returns
- `x`: Final solution vector
- `f`: Final residual vector at solution
- `g`: Final gradient vector at solution  
- `iter`: Number of iterations performed

# Algorithm Details
The method uses a trust region approach where at each iteration:
1. Solves a bounded quadratic subproblem using `solve_bounded_subproblemm`
2. Evaluates the reduction ratio between actual and predicted cost reduction
3. Updates the trust region radius based on the reduction ratio
4. Accepts or rejects the step based on sufficient reduction criteria

Convergence is achieved when either:
- Gradient norm falls below `gtol`
- Function reduction falls below `ftol * current_cost`
- Trust region radius falls below `min_trust_radius`
- Maximum iterations reached

# Notes
- The initial trust region radius is automatically set to `||x0||` if `||x0|| > 1e-4`
- The method handles box constraints through the active set strategy in the subproblem solver
- Progress information is printed during iterations for monitoring convergence
"""

function active_set_svd_trust_region(
    res::Function, jac::Function,
    x0::Array{T};
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    step_threshold::Real = 0.01,
    shrink_threshold::Real = 0.25,
    expand_threshold::Real = 0.75,
    shrink_factor::Real = 0.25,
    expand_factor::Real = 2.0,
    max_iter::Int = 100,
    gtol::Real = 1e-6,
    ftol::Real = 1e-15,
    τ::Real = 1e-12,
    ) where T

    # Initialize
    x = copy(x0)
    f = res(x)
    J = jac(x)
    cost = 0.5 * dot(f, f)
    g = J' * f
    if norm(x0) > 1e-4
        initial_radius = norm(x0)
    end
    radius = initial_radius
    # Check initial convergence
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    #iterations
    for iter in 1:max_iter
        λ, δ = solve_bounded_subproblemm(J, f, radius, lb - x, ub - x, 20)
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        # Predicted reduction using QR factorization
        Jδ = J * δ
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = 0.5*dot(Jδ, Jδ)+ λ*dot(δ,δ)
        if predicted_reduction <= 0 #this potentially means the δ is wrong but we leave some margin
            println("Non-positive predicted reduction, shrinking radius")
            radius *= shrink_factor
            continue
        end
        # the reduction ratio
        ρ = actual_reduction / predicted_reduction
        # Update trust region radius
        if (ρ >= expand_threshold) && (λ > 0)
            radius = min(max_trust_radius, expand_factor * norm(δ))
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        # Accept or reject step
        if ρ >= step_threshold
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = J' * f
            println("Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius")
            # Check convergence
            if norm(g, 2) < gtol
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            if actual_reduction < ftol * cost
                println("Function tolerance criterion reached")
                return x, f, g, iter
            end
        else
            println("Step rejected, ρ = $ρ")
        end
        # Check trust region size
        if radius < min_trust_radius
            println("Trust region radius below minimum")
            return x, f, g, iter
        end        
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end


