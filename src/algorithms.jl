function lm_trust_region(
    res::Function,
    jac::Function,
    x0::Array{T},
    subproblem_strategy::SubProblemStrategy = SVDSolve(),
    scaling_strategy::ScalingStrategy = NoScaling();
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
    ftol::Real = 1e-15,
    norm_overrides_initial_radius::Bool = true,
) where {T}

    # Initialize
    x = copy(x0)
    x_trial = copy(x0) # Buffer for candidate step
    f = res(x)
    J = jac(x)
    Jδ = Vector{T}(undef, length(f))
    cost = 0.5 * dot(f, f)
    g = J' * f
    cache = SubproblemCache(subproblem_strategy, scaling_strategy, J)
    if norm_overrides_initial_radius && norm(x0) > 1e-4
        initial_radius = norm(cache.scaling_matrix * x0)
    end
    radius = initial_radius
    # Check initial convergence
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    #iterations
    for iter = 1:max_iter
        # Compute step using QR-facorization
        λ, δ = solve_subproblem(subproblem_strategy, J, f, radius, cache)
        # Evaluate new point
        @. x_trial = x + δ
        f_new = res(x_trial)
        cost_new = 0.5 * dot(f_new, f_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        # Predicted reduction using QR factorization
        mul!(Jδ, J, δ)
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = 0.5*dot(Jδ, Jδ)+λ*dot(δ, δ)
        if predicted_reduction <= 0 #this potentially means the δ is wrong but we leave some margin
            println("Non-positive predicted reduction, shrinking radius")
            radius *= shrink_factor
            continue
        end
        # the reduction ratio
        ρ = actual_reduction / predicted_reduction
        # Update trust region radius
        if (ρ >= expand_threshold) && (λ > 0)
            radius = min(max_trust_radius, expand_factor * radius)
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        # Accept or reject step
        if ρ >= step_threshold
            @. x = x_trial
            @. f = f_new
            cost = cost_new
            J = jac(x)
            mul!(g, J', f)
            println(
                "Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius",
            )
            # Check convergence
            if norm(g, 2) < gtol
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            if actual_reduction < ftol * max(cost, 1.0)
                println("Function tolerance criterion reached")
                return x, f, g, iter
            end
            # update cache
            factorize!(cache, subproblem_strategy, J)
            cache.scaling_matrix .= scaling(scaling_strategy, J)
            # To verify allocations reduced, check cache.J_buffer
            if cache.J_buffer === nothing
                @warn "Optimized path NOT taken: J is $(typeof(J))"
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
