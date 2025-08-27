function lm_trust_region(
    res::Function, jac::Function,
    x0::Array{T};
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
        # Compute step using QR-facorization
        δgn = J \ -f
        if norm(δgn) <= radius
            # The gauss newton minimal-norm step that fits the model is within the radius.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ(radius, J, f, 100)
        end
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


function lm_trust_region_scaled(
    res::Function, jac::Function,
    x0::Array{T};
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
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    m,n = size(J)
    Dk = Diagonal(ones(eltype(x0), n))
    column_norms = [norm(J[:, i]) for i in axes(J, 2)]
    for i in axes(J, 2)
        Dk[i, i] = max(τ, column_norms[i])
    end
    δgn = J \ -f
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter in 1:max_iter
        # Compute step using QR-based trust region
        m, n = size(J)
        if iter > 1
            δgn = J \ -f # avoid extra computation
        end
        if norm(Dk * δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ_scaled(radius, J, Dk, f, 100)
        end
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        # Predicted reduction using QR factorization
        Jδ = J * δ
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = 0.5*dot(Jδ, Jδ)+ λ*dot(Dk * δ, Dk * δ)
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
            column_norms = [norm(J[:, i]) for i in axes(J, 2)]
            for i in axes(J, 2)
                Dk[i, i] = max(Dk[i, i], column_norms[i])
            end
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

# Uses and reuses the svd factorization, thus useful when calculating the step constrained by λ!
function lm_svd_trust_region(
    res::Function, jac::Function,
    x0::Array{T};
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
    svdls = svd(J)
    #iterations
    for iter in 1:max_iter
        # Compute step using QR-based trust region
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
            svdls = svd(J)  # Update SVD after step
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