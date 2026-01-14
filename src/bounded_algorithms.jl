"""
    lm_trust_region_reflective(
        res::Function,
        jac::Function,
        x0::Array{T},
        subproblem_strategy::SubProblemStrategy = SVDSolve(),
        scaling_strategy::ScalingStrategy = ColemanandLiScaling();
        lb::Array{T} = fill(-Inf, length(x0)),
        ub::Array{T} = fill(Inf, length(x0)),
        initial_radius::Real = 1.0,
        max_trust_radius::Real = 1e12,
        min_trust_radius::Real = 1e-8,
        step_threshold::Real = 0.001,
        shrink_threshold::Real = 0.25,
        expand_threshold::Real = 0.75,
        shrink_factor::Real = 0.25,
        expand_factor::Real = 2.0,
        max_iter::Int = 100,
        gtol::Real = 1e-6,
        norm_overrides_initial_radius::Bool = true,
    ) where {T}

Solves a bounds-constrained nonlinear least squares problem using a trust-region reflective algorithm.
This method is particularly suitable for problems where the variables `x` are constrained within
lower (`lb`) and upper (`ub`) bounds. It incorporates Coleman and Li's reflective strategy
to handle boundaries and approximates the subproblem with a Levenberg-Marquardt-like trust region.

# Arguments
- `res::Function`: The residual function `f(x)` where `f` is a vector-valued function
                   and the objective is to minimize `sum(f(x).^2)`.
- `jac::Function`: The Jacobian function `J(x)` which returns the Jacobian matrix `J` of `res(x)`.
- `x0::Array{T}`: The initial guess for the solution vector `x`.
- `subproblem_strategy::SubProblemStrategy`: Strategy for solving the trust-region subproblem (e.g., `SVDSolve()`, `QRSolve()`).
                                           Defaults to `SVDSolve()`.
- `scaling_strategy::ScalingStrategy`: Strategy for scaling the variables, particularly important
                                     for the reflective method. Defaults to `ColemanandLiScaling()`.

# Keywords
- `lb::Array{T}`: Lower bounds for `x`. Defaults to `-Inf` for all variables.
- `ub::Array{T}`: Upper bounds for `x`. Defaults to `Inf` for all variables.
- `initial_radius::Real`: The initial trust region radius. Defaults to `1.0`.
- `max_trust_radius::Real`: The maximum allowed trust region radius. Defaults to `1e12`.
- `min_trust_radius::Real`: The minimum allowed trust region radius. If the radius
                            falls below this, the optimization terminates. Defaults to `1e-8`.
- `step_threshold::Real`: Threshold for accepting a step. If actual reduction / predicted reduction
                          is below this, the step is rejected. Defaults to `0.001`.
- `shrink_threshold::Real`: If the ratio of actual to predicted reduction is less than this,
                            the trust region radius is shrunk. Defaults to `0.25`.
- `expand_threshold::Real`: If the ratio of actual to predicted reduction is greater than or equal
                            to this, and `λ > 0`, the trust region radius is expanded. Defaults to `0.75`.
- `shrink_factor::Real`: Factor by which to shrink the trust region radius. Defaults to `0.25`.
- `expand_factor::Real`: Factor by which to expand the trust region radius. Defaults to `2.0`.
- `max_iter::Int`: Maximum number of iterations. Defaults to `100`.
- `gtol::Real`: Gradient tolerance. The algorithm terminates if the norm of the gradient
                falls below this value. Defaults to `1e-6`.
- `norm_overrides_initial_radius::Bool`: If `true`, the `initial_radius` is set to the norm
                                        of the scaled Gauss-Newton step. Defaults to `true`.

# Returns
- `x::Array{T}`: The optimized solution vector.
- `f::Array{T}`: The residuals `res(x)` at the optimized `x`.
- `g::Array{T}`: The gradient `J(x)' * f(x)` at the optimized `x`.
- `iter::Int`: The number of iterations performed.
"""
function lm_trust_region_reflective(
    res::Function,
    jac::Function,
    x0::Array{T},
    subproblem_strategy::SubProblemStrategy = SVDSolve(),
    scaling_strategy::ScalingStrategy = ColemanandLiScaling();
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    step_threshold::Real = 0.001,
    shrink_threshold::Real = 0.25,
    expand_threshold::Real = 0.75,
    shrink_factor::Real = 0.25,
    expand_factor::Real = 2.0,
    max_iter::Int = 100,
    gtol::Real = 1e-6,
    norm_overrides_initial_radius::Bool = true,
) where {T}
    # Initialize
    x = copy(x0)
    f = res(x)
    J = jac(x)
    cost = 0.5 * dot(f, f)
    g = J' * f
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    # check x0 is within bounds and restrict it to bounds
    #x = clamp.(x, lb, ub)
    m, n = size(J)
    if isa(scaling_strategy, ColemanandLiScaling)
        cache = ColemanandLiCache(subproblem_strategy, scaling_strategy, J; x=x, lb=lb, ub=ub, g=g)
    else
        cache = SubproblemCache(subproblem_strategy, scaling_strategy, J; x=x, lb=lb, ub=ub, g=g)
    end
    # Dk, A, v = affine_scale_matrix(x, lb, ub, g)
    Dk = cache.scaling_matrix
    A = cache.Jv
    δgn = [J; √A*Dk] \ [-f; zeros(n)]
    if norm_overrides_initial_radius
        initial_radius = norm(Dk*δgn)
    end
    radius = initial_radius
    for iter = 1:max_iter
        # Compute step using trust region strategy
        λ, δ = solve_subproblem(subproblem_strategy, J, f, radius, cache)
        # if iter > 1
        #     δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation
        # end
        # if norm(Dk * δgn) <= radius
        #     # The minimal-norm step that perfectly fits the model is within the radius.
        #     # This is the ideal solution. There is no need to make the step longer.
        #     δ = δgn
        #     λ = 0.0
        # else
        #     # The smallest "perfect" step is too big. We must find a damped
        #     # step on the boundary using the standard LM approach.
        #     λ, δ = find_λ_scaled_b(radius, J, A, Dk, f, 100)
        # end
        δ, Ψ = bounded_step(scaling_strategy, δ, lb, ub, Dk, A, J, g, x, radius)

        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        numerator = cost_new - cost #+ 1/2*δ'*Cₖ*δ
        if numerator > 0
            radius *= shrink_factor
            println("step rejected - positive update")
        else
            #Ψ = g'*δ + 1/2*(δ'* (J'J+ Cₖ) * δ) #
            ρᶠ = numerator / Ψ # definition of the step improvement
            # Accept or reject step
            if ρᶠ > step_threshold
                x .= x_new
                f .= f_new
                cost = cost_new
                J .= jac(x)
                g .= J' * f
                if isa(scaling_strategy, ColemanandLiScaling)
                    cache = ColemanandLiCache(subproblem_strategy, scaling_strategy, J; x=x, lb=lb, ub=ub, g=g)
                else
                    cache = SubproblemCache(subproblem_strategy, scaling_strategy, J; x=x, lb=lb, ub=ub, g=g)
                end
                println(
                    "Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius",
                )
                # Check convergence
                if norm(g, 2) < gtol
                    println("Gradient convergence criterion reached")
                    return x, f, g, iter
                end
                # if actual_reduction < ftol * cost
                #     println("Function tolerance criterion reached")
                #     return x, f, g, iter
                # end
                # Update trust region radius
                if (ρᶠ >= expand_threshold) && (λ > 0)
                    radius = min(max_trust_radius, expand_factor * radius)
                end
            else
                println("Step rejected, ρᶠ = $ρᶠ")
            end
            if (ρᶠ < shrink_threshold) || isnan(ρᶠ)
                radius *= shrink_factor
            end
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
