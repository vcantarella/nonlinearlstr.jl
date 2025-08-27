
"""
    qr_nlss_bounded_trust_region(res::Function, jac::Function, x0, lb, ub; kwargs...)

QR-based nonlinear least squares solver with bounded trust region method.
Uses QR factorization for improved numerical stability.

# Arguments
- `res::Function`: Residual function r(x) returning vector of residuals
- `jac::Function`: Jacobian function J(x) returning m×n matrix  
- `x0::Array`: Initial parameter guess
- `lb::Array`: Lower bounds
- `ub::Array`: Upper bounds

# Keyword Arguments
- `initial_radius::Real = 1.0`: Initial trust region radius
- `max_trust_radius::Real = 100.0`: Maximum trust region radius
- `min_trust_radius::Real = 1e-12`: Minimum trust region radius
- `step_threshold::Real = 0.01`: Threshold for accepting steps
- `shrink_threshold::Real = 0.25`: Threshold for shrinking radius
- `expand_threshold::Real = 0.9`: Threshold for expanding radius
- `shrink_factor::Real = 0.25`: Factor for shrinking radius
- `expand_factor::Real = 2.0`: Factor for expanding radius
- `max_iter::Int = 100`: Maximum iterations
- `gtol::Real = 1e-6`: Gradient tolerance
- `ftol::Real = 1e-15`: Function tolerance

# Returns
- Tuple (x, residuals, gradient, iterations)
"""
function qr_nlss_trust_region_v2(
    res::Function, jac::Function,
    x0::Array{T};
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 100.0,
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
    # regularization::Real = 0.0,
    ) where T

    # Initialize
    x = copy(x0)
    f = res(x)
    J = jac(x)
    qrls = qr(J, ColumnNorm())
    
    cost = 0.5 * dot(f, f)
    g = compute_gradient(J, f)
    
    radius = initial_radius
    
    # Check initial convergence
    if norm(g, Inf) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end

    D = Diagonal(zeros(eltype(x0), length(x0)))
    D_inv = copy(D)

    for iter in 1:max_iter
        # Compute step using QR-based trust region
        qrls = qr(J, ColumnNorm())

        for i in axes(J, 2)
            D[i, i] = maximum([τ, norm(J[:, i]), D[i, i]])
            D_inv[i, i] = 1 / D[i, i]
        end
        J_scaled = J * D_inv
        δgn, hardcase = solve_gauss_newton(J_scaled, f)
        λ = 0.0
        if norm(δgn) < radius
            δgn = D_inv* δgn
            #δgn = clamp(δgn, lb - x, ub - x)
            δ = δgn
        else
            λ, δ = find_λ!(radius, J_scaled, f, 100)
            δ = D_inv * δ
            #δ = clamp(δ, lb - x, ub - x)
        end
        
        if norm(δ) < min_trust_radius
            println("Step size below minimum trust radius")
            return x, f, g, iter
        end
        
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        
        # Predicted reduction using QR factorization
        # pred = -g'δ - 0.5 δ'(J'J)δ
        Jδ = J * δ
        if λ > 0
            predicted_reduction = 1/2*norm(Jδ)^2 * λ*norm(D*δ)^2
        else
            predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        end
        
        #predicted_reduction =  -(0.5 * sk' * Bk * sk + g' * sk)
        
        if predicted_reduction <= 0
            if actual_reduction > ftol*cost
                x = x_new
                f = f_new
                cost = cost_new
                J = jac(x)
                g = compute_gradient(J, f)
            else
                println("Non-positive predicted reduction, shrinking radius")
                radius *= shrink_factor
            end
            continue
        end
        
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
            g = compute_gradient(J, f)
            
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


"""
    qr_nlss_trust_region(res::Function, jac::Function, x0, lb, ub; kwargs...)

QR-based nonlinear least squares solver with bounded trust region method.
Uses QR factorization for improved numerical stability.

# Arguments
- `res::Function`: Residual function r(x) returning vector of residuals
- `jac::Function`: Jacobian function J(x) returning m×n matrix  
- `x0::Array`: Initial parameter guess
- `lb::Array`: Lower bounds
- `ub::Array`: Upper bounds

# Keyword Arguments
- `initial_radius::Real = 1.0`: Initial trust region radius
- `max_trust_radius::Real = 100.0`: Maximum trust region radius
- `min_trust_radius::Real = 1e-12`: Minimum trust region radius
- `step_threshold::Real = 0.01`: Threshold for accepting steps
- `shrink_threshold::Real = 0.25`: Threshold for shrinking radius
- `expand_threshold::Real = 0.9`: Threshold for expanding radius
- `shrink_factor::Real = 0.25`: Factor for shrinking radius
- `expand_factor::Real = 2.0`: Factor for expanding radius
- `max_iter::Int = 100`: Maximum iterations
- `gtol::Real = 1e-6`: Gradient tolerance
- `ftol::Real = 1e-15`: Function tolerance

# Returns
- Tuple (x, residuals, gradient, iterations)
"""
function qr_bounded_nlss_trust_region(
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
    qrls = qr(J, ColumnNorm())
    
    cost = 0.5 * dot(f, f)
    g = compute_gradient(J, f)

    if norm(x0) > 1e-4
        initial_radius = norm(x0)
    end

    
    radius = initial_radius
    
    # Check initial convergence
    if norm(g, Inf) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end

    for iter in 1:max_iter
        # Compute step using QR-based trust region
        #qrls = qr(J)

        m, n = size(J)
        δgn, is_rank_deficient, F = solve_gauss_newton_v3(J, f)
        on_boundary = false
        λ = 0.0

        if m < n
            # --- UNDERDETERMINED CASE (More parameters than residuals) ---
            if norm(δgn) <= radius
                # The minimal-norm step that perfectly fits the model is within the radius.
                # This is the ideal solution. There is no need to make the step longer.
                δ = δgn
                on_boundary = (abs(norm(δgn) - radius) < 1e-9)
            else
                # The smallest "perfect" step is too big. We must find a damped
                # step on the boundary using the standard LM approach.
                λ, δ = find_λ!(radius, J, f, 100)
                on_boundary = true
            end
        else
            # --- SQUARE OR OVERDETERMINED CASE (m >= n) ---
            if norm(δgn) <= radius
                if !is_rank_deficient
                    # Full rank, standard Gauss-Newton step is optimal.
                    δ = δgn
                    on_boundary = (abs(norm(δgn) - radius) < 1e-9)
                else
                    # This is the TRUE "Hard Case" for m >= n systems.
                    on_boundary = true
                    
                    # Extract the null space vector from the SVD we already computed.
                    tol = maximum(size(J)) * eps(F.S[1])
                    effective_rank = count(s -> s > tol, F.S)
                    null_space_idx = effective_rank + 1
                    z = F.V[:, null_space_idx]
                    
                    # Solve for α to extend the step to the boundary.
                    a = dot(z, z)
                    b = 2 * dot(δgn, z)
                    c = dot(δgn, δgn) - radius^2
                    
                    discriminant = b^2 - 4*a*c
                    if discriminant < 0
                        println("Warning: Negative discriminant in hard case. Using GN step.")
                        δ = δgn # Fallback
                    else
                        # Deterministically choose the positive root for α.
                        α = (-b + sqrt(discriminant)) / (2a)
                        δ = δgn + α * z
                    end
                end
            else
                # Standard case: Gauss-Newton step is too long, find damped LM step.
                λ, δ = find_λ!(radius, J, f, 100)
                on_boundary = true
            end
        end

        # δgn, z, hard_case = solve_gauss_newton_v2(J, f)
        # on_boundary = false

        # if norm(δgn) < radius
        #     if !hard_case
        #         δ = δgn
        #         λ = 0.0
        #         on_boundary = false
        #     else #hard case!
        #         # find α: ||δgn||² + 2α(δgn'z) + α²||z||² = Δ²
        #         δgn2 = dot(δgn, δgn)
        #         δgnz = dot(δgn, z)
        #         z2 = dot(z, z)
        #         a = z2
        #         b = 2 * δgnz
        #         c = δgn2 - radius^2
        #         # Solve quadratic equation: aα² + bα + c = 0
        #         discriminant = b^2 - 4a*c
        #         if discriminant < 0
        #             println("No valid step found")
        #             radius *= shrink_factor
        #             continue
        #         end
        #         # α₁ = (-b - sqrt(discriminant)) / (2a)
        #         α₂ = (-b + sqrt(discriminant)) / (2a)
        #         # Calculate the two potential steps
        #         # δ₁ = δgn + α₁ * z
        #         δ₂ = δgn + α₂ * z

        #         # Evaluate the model at each point
        #         # Note: g = J'f, and J'Jδ = J'(Jδ)
        #         # q₁ = -dot(g, δ₁) - 0.5 * dot(J*δ₁, J*δ₁)
        #         # q₂ = -dot(g, δ₂) - 0.5 * dot(J*δ₂, J*δ₂)

        #         # Choose the step that minimizes the model
        #         # if q₁ >= q₂
        #         #     δ = δ₁
        #         # else
        #         #     
        #         # end
        #         δ = δ₂
        #         on_boundary = true
        #     end
            
        # else
        #     # λ = find_λ!(λ₀, radius, J, f, max_iter)
        #     # # println("Iteration: $iter, λ: $λ, radius: $radius")
        #     # δ = qr_trust_region_step(J, qrls, f, radius, lb, ub, x; λ=λ)
        #     λ, δ = find_λ!(radius, J, f, 100)
        #     on_boundary = true
        #     # δ = clamp(δ, lb - x, ub - x)
        # end
        
        if norm(δ) < min_trust_radius
            println("Step size below minimum trust radius")
            return x, f, g, iter
        end
        
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        
        # Predicted reduction using QR factorization
        Jδ = J * δ
        predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        # predicted_reduction = 1/2*norm(Jδ)^2 + λ*norm(δ)^2

        #predicted_reduction =  -(0.5 * sk' * Bk * sk + g' * sk)
        
        if predicted_reduction <= 0
            if actual_reduction > ftol*cost
                x = x_new
                f = f_new
                cost = cost_new
                J = jac(x)
                g = compute_gradient(J, f)
            else
                println("Non-positive predicted reduction, shrinking radius")
                radius *= shrink_factor
            end
            continue
        end
        
        ρ = actual_reduction / predicted_reduction
        
        # Update trust region radius
        if (ρ >= expand_threshold) && on_boundary
            radius = min(max_trust_radius, max(radius, expand_factor * norm(δ)))
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        
        # Accept or reject step
        if ρ >= step_threshold
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = compute_gradient(J, f)
            
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

function bounded_gauss_newton(
    res::Function, jac::Function,
    x0::Array{T}, lb::Array{T}, ub::Array{T};
    max_iter::Int = 100,
    inner_max_iter::Int = 100,
    gtol::Real = 1e-6,
    ) where T

    # Initialize variables
    x = copy(x0)
    f = res(x)
    J = jac(x)
    g = J'f
    cost = 0.5 * dot(f, f)
    for iter in 1:max_iter
        # Check convergence
        if norm(g, 2) < gtol
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
        # Bounded Newton step
        δ = lsq_box(J, -f, lb - x, ub - x; maxiter=inner_max_iter)
        x += δ
        f = res(x)
        cost = 0.5 * dot(f, f)
        J = jac(x)
        g = J'f
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end


function bounded_trust_region(
    res::Function, jac::Function,
    x0::Array{T}, lb::Array{T}, ub::Array{T};
    max_iter::Int = 100,
    inner_max_iter::Int = 100,
    gtol::Real = 1e-6,
    ) where T

    # Trust region parameters
    Δ₀ = 1.0 #initial radius - Wang & Yuan 2013
    Δ_m = 100.0 #max radius - Wang & Yuan 2013
    η₁ = 1e-8 #step threshold - Nocedal & Wright 2007
    η₂ = 0.25 #shrink threshold - Nocedal & Wright 2007
    η₃ = 0.75 #expand threshold - Nocedal & Wright 2007
    # Initialize variables
    x = copy(x0)
    f = res(x)
    J = jac(x)
    Δ = Δ₀
    g = J'f
    cost = 0.5 * dot(f, f)
    
    affine_cache = (
        Dk=Diagonal(ones(length(x))),
        inv_Dk=Diagonal(ones(length(x))),
        ak=ones(length(x)),
        bk=ones(length(x))
    )
    update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
    Dk, inv_Dk, ak, bk = affine_cache
    g_hat = Dk*g
    J_hat = J*Dk
    lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
    hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
    for iter in 1:max_iter
        # Check convergence
        if norm(g, 2) < gtol
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
        
        # Bounded Newton step
        δ = lsq_box(J_hat, -f, lo_step, hi_step; maxiter=inner_max_iter)
        println("Iteration: $iter, norm(δ): $(norm(δ, 2)), current radius: $Δ")
        predicted_reduction = -dot(g_hat, δ) - 0.5 * dot(J_hat*δ, J_hat*δ)
        δ = Dk * δ # scale back to original space
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        actual_reduction = cost - cost_new
        ρ = actual_reduction/predicted_reduction
        if ρ < η₂
            Δ *= η₂
            update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
            g_hat = Dk*g
            J_hat = J*Dk
            lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
            hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
        end
        if ρ >= η₁
            # Accept step
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = J'f
            println("Accepted step, new cost: $cost, norm(g): $(norm(g, 2))")       
            # Update trust region radius
            if ρ >= η₃
                Δ = min(Δ_m, max(Δ, 2 * norm(δ)))
                println("Expanded trust region to $Δ")
            end
            update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
            g_hat = Dk*g
            J_hat = J*Dk
            lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
            hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
        else
            println("Rejected step, ρ = $ρ")
        end
        # After update_Dk!
        if any(isnan, diag(Dk)) || any(isinf, diag(Dk)) || any(isnan, diag(inv_Dk)) || any(isinf, diag(inv_Dk))
            error("NaN or Inf detected in scaling matrices Dk or inv_Dk")
        end

    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end


function fake_trust_region_reflective(
    res::Function, jac::Function,
    x0::Array{T}, lb::Array{T}, ub::Array{T};
    max_iter::Int = 100,
    inner_max_iter::Int = 100,
    gtol::Real = 1e-6,
    ) where T

    # Trust region parameters
    Δ₀ = 1.0 #initial radius - Wang & Yuan 2013
    Δ_m = 100.0 #max radius - Wang & Yuan 2013
    η₁ = 1e-8 #step threshold - Nocedal & Wright 2007
    η₂ = 0.25 #shrink threshold - Nocedal & Wright 2007
    η₃ = 0.75 #expand threshold - Nocedal & Wright 2007
    # Initialize variables
    x = copy(x0)
    f = res(x)
    J = jac(x)
    Δ = Δ₀
    g = J'f
    cost = 0.5 * dot(f, f)
    
    affine_cache = (
        Dk=Diagonal(ones(length(x))),
        inv_Dk=Diagonal(ones(length(x))),
        ak=ones(length(x)),
        bk=ones(length(x))
    )
    update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
    Dk, inv_Dk, ak, bk = affine_cache
    g_hat = Dk*g
    J_hat = J*Dk
    lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
    hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
    for iter in 1:max_iter
        # Check convergence
        if norm(g, 2) < gtol
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
        
        # Bounded Newton step
        # for numerical stability we take a damped gauss newton step as our
        # first try:
        λ = 1e-6
        δ = qr_regularized_solve_scaled(J, inv_Dk, -f, λ)
        if norm(inv_Dk*δ) > Δ
            # Then we need to calculate the regularized solve:
            λ, δ = find_λ_scaled!(Δ, J, inv_Dk, f, 100)
        end
        # Now we check for the bounds:
        if any(x + δ .< lb) || any(x + δ .> ub)
            # Now we need to decide between three steps
            # Reflect the step back into bounds
            α, hits = step_size_to_bound(x, δ, lb, ub)
            println("Reflective step, α: $α, hits: $hits")
            if α < 1.0
                δ *= α
            end
        end
        println("Iteration: $iter, norm(δ): $(norm(δ, 2)), current radius: $Δ")
        predicted_reduction = -dot(g, δ) - 0.5 * dot(J*δ, J*δ)
        if predicted_reduction <= 0
            println("Non-positive predicted reduction, shrinking radius")
            Δ *= η₂
            update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
            continue  # Try again with smaller radius
        end
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        actual_reduction = cost - cost_new
        ρ = actual_reduction/predicted_reduction
        if ρ < η₂
            Δ *= η₂
            update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
            g_hat = Dk*g
            J_hat = J*Dk
            lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
            hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
        end
        if ρ >= η₁
            # Accept step
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = J'f
            println("Accepted step, new cost: $cost, norm(g): $(norm(g, 2))")       
            # Update trust region radius
            if ρ >= η₃
                Δ = min(Δ_m, max(Δ, 2 * norm(δ)))
                println("Expanded trust region to $Δ")
            end
            update_Dk!(affine_cache, x, lb, ub, g, Δ, 1e-10)
            g_hat = Dk*g
            J_hat = J*Dk
            lo_step = max.(inv_Dk * (lb - x), -Δ*ones(length(x)))
            hi_step = min.(inv_Dk * (ub - x), Δ*ones(length(x)))
        else
            println("Rejected step, ρ = $ρ")
        end
        # After update_Dk!
        if any(isnan, diag(Dk)) || any(isinf, diag(Dk)) || any(isnan, diag(inv_Dk)) || any(isinf, diag(inv_Dk))
            error("NaN or Inf detected in scaling matrices Dk or inv_Dk")
        end

    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end


function lm_trust_region_v2(
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
    qrls = qr(J, ColumnNorm())
    #iterations
    for iter in 1:max_iter
        # Compute step using QR-based trust region
        δgn = qrls \ -f
        if norm(δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ_v2!(radius, qrls, f, 100)
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
            qrls = qr(J, ColumnNorm())
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


function bounded_trust_region(
    func::Function, grad::Function,
    hess::Function, x0::Array{T},
    lb::Array{T}, ub::Array{T};
    max_trust_radius = nothing,
    min_trust_radius = 1e-12,
    initial_radius = 1.0,
    step_threshold = 0.01,
    shrink_threshold = 0.25,
    expand_threshold = 0.9,
    shrink_factor = 0.25,
    expand_factor = 2.0,
    Beta = 0.99999,
    max_iter = 100,
    gtol = 1e-6,
    ) where T
    # Check if x0 is within bounds
    if any(x0 .< lb) || any(x0 .> ub)
        error("Initial guess x0 is not within bounds")
    end

    f0 = func(x0)
    if max_trust_radius === nothing
        max_radius = max(norm(f0), maximum(x0) - minimum(x0))
    else
        max_radius = max_trust_radius
    end
    

    g0 = grad(x0)

    Bk = hess(x0)

    #Step 1 termination test:
    if (norm(g0) < gtol) || (initial_radius < min_trust_radius)
        println("Initial guess is already a minimum gradientwise")
        return x0, f0, g0, 0
    end

    ## initializing the scaling variables:
    ak = zeros(eltype(x0), length(x0))
    bk = zeros(eltype(x0), length(x0))
    Dk = Diagonal(ones(eltype(x0), length(x0)))
    inv_Dk = copy(Dk)
    affine_cache = (ak = ak, bk = bk, Dk = Dk, inv_Dk = inv_Dk)

    # initializing the x and g vectors
    x = x0
    g = g0
    f = f0
    radius = initial_radius

    for iter in 1:max_iter
        #Step 2: Determine trial step
        ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
        ## Step 2.1 Calculate the scaling vectors and the scaling matrix
        update_Dk!(affine_cache, x, lb, ub, g, radius, 1e-16)
        Dk = affine_cache.Dk
        inv_Dk = affine_cache.inv_Dk
        ## Step 2.2: Solve the trust region subproblem in the scaled space: d_hat = inv(D)*d
        # f_dhat(d_hat) = (Dk*g)'*d_hat + 0.5*d_hat'*(Dk*Bk*Dk)*d_hat
        dhatl = inv_Dk * (lb - x)
        dhatu = inv_Dk * (ub - x)
        A = Dk*Bk*Dk
        b = Dk*g
        d_hat = zeros(eltype(x0), length(x0))
        try
            d_hat = tcg(A, b, radius, dhatl, dhatu, gtol, 1000)
        catch e
            if isa(e, DomainError)
                println("Domain error encountered: ", e)
                radius = 0.5 * radius
                if (radius < min_trust_radius) || (norm(g) < gtol)
                    # print gradient convergence
                    println("Gradient convergence criterion reached")
                    return x, f, g, iter
                end
                continue
            else
                rethrow(e)
            end
        end
        sk = Beta .* Dk * d_hat

        if norm(sk) < gtol
            println("Step size convergence criterion reached")
            return x, f, g, iter
        end

        f_new = func(x+sk)
        actual_reduction = f - f_new
        if actual_reduction < 0
            radius = 0.5 * radius
            if (radius < min_trust_radius) || (norm(g) < gtol)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            continue
        end
        pred_reduction = -(0.5 * sk' * Bk * sk + g' * sk)
        if pred_reduction < 0
            radius = 0.5 * radius
            if (radius < min_trust_radius) || (norm(g) < gtol)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            continue
        end
        ρ = actual_reduction / pred_reduction

        if ρ ≥ step_threshold
            x = x + sk
            f = f_new
            g = grad(x)
            Bk = hess(x)
            println("Iteration: $iter, f: $f, norm(g): $(norm(g))")
            println("--------------------------------------------")
        #Step 4: Update the trust region radius
            if ρ > expand_threshold
                radius = maximum([radius, expand_factor * norm(inv_Dk*sk)])
                if radius > max_radius
                    radius = max_radius
                end
            elseif ρ < shrink_threshold
                maximum([0.5*radius, shrink_factor * norm(inv_Dk*sk)])
            end
        else
            radius = 0.5 * radius
        end
        if (radius < min_trust_radius) || (norm(g) < gtol)
            # print gradient convergence
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end


function nlss_bounded_trust_region(
    res::Function, jac::Function,
    x0::Array{T}, lb::Array{T}, ub::Array{T};
    max_trust_radius = nothing,
    min_trust_radius = 1e-12,
    initial_radius = 1.0,
    step_threshold = 0.01,
    shrink_threshold = 0.25,
    expand_threshold = 0.9,
    shrink_factor = 0.25,
    expand_factor = 2.0,
    Beta = 0.99999,
    max_iter = 100,
    gtol = 1e-6,
    ) where T
    # Check if x0 is within bounds
    if any(x0 .< lb) || any(x0 .> ub)
        error("Initial guess x0 is not within bounds")
    end

    f0 = res(x0)
    if max_trust_radius === nothing
        max_radius = max(norm(f0), maximum(x0) - minimum(x0))
    else
        max_radius = max_trust_radius
    end
    
    J0 = jac(x0)
    g0 = J0'f0

    #Step 1 termination test:
    if (norm(g0) < gtol) || (initial_radius < min_trust_radius)
        println("Initial guess is already a minimum gradientwise")
        return x0, f0, g0, 0
    end

    ## initializing the scaling variables:
    ak = zeros(eltype(x0), length(x0))
    bk = zeros(eltype(x0), length(x0))
    Dk = Diagonal(ones(eltype(x0), length(x0)))
    inv_Dk = copy(Dk)
    affine_cache = (ak = ak, bk = bk, Dk = Dk, inv_Dk = inv_Dk)

    # initializing the x and g vectors
    x = x0
    g = g0
    f = f0
    J = J0
    radius = initial_radius

    for iter in 1:max_iter
        #Step 2: Determine trial step
        ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
        ## Step 2.1 Calculate the scaling vectors and the scaling matrix
        update_Dk!(affine_cache, x, lb, ub, g, radius, 1e-16)
        Dk = affine_cache.Dk
        inv_Dk = affine_cache.inv_Dk
        ## Step 2.2: Solve the trust region subproblem in the scaled space: d_hat = inv(D)*d
        # f_dhat(d_hat) = (Dk*g)'*d_hat + 0.5*d_hat'*(Dk*Bk*Dk)*d_hat
        dhatl = inv_Dk * (lb - x)
        dhatu = inv_Dk * (ub - x)
        J = J*Dk
        d_hat = zeros(eltype(x0), length(x0))
        try
            d_hat = tcgnlss(f,J, radius, dhatl, dhatu, gtol, 1000)
        catch e
            if isa(e, DomainError)
                println("Domain error encountered: ", e)
                radius = 0.5 * radius
                if (radius < min_trust_radius) || (norm(g) < gtol)
                    # print gradient convergence
                    println("Gradient convergence criterion reached")
                    return x, f, g, iter
                end
                continue
            else
                rethrow(e)
            end
        end
        sk = Beta .* Dk * d_hat

        if norm(sk) < gtol
            println("Step size convergence criterion reached")
            return x, f, g, iter
        end

        f_new = res(x+sk)
        actual_reduction = 0.5*(sum(f'f) - sum(f_new'f_new))
        if actual_reduction < 0
            radius = 0.5 * radius
            if (radius < min_trust_radius) || (norm(g) < gtol)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end
            continue
        end
        pred_reduction = -(0.5 * sk' * J'*(J * sk) + sk'g)
        actual_reduction = 1/2(norm(f'f) - norm(f_new'f_new))
        #pred_reduction = -(0.5 * sk' * H.Q * H.R * sk + g' * sk)
        ρ = actual_reduction / pred_reduction

        if ρ >= step_threshold
            x = x + sk
            f = f_new
            J = jac(x)
            g = J' * f
            println("Iteration: $iter, f: $(0.5*sum(f'f)), norm(g): $(norm(g))")
            println("--------------------------------------------")
        #Step 4: Update the trust region radius
            if ρ > expand_threshold
                radius = maximum([radius, expand_factor * norm(inv_Dk*sk)])
                if radius > max_radius
                    radius = max_radius
                end
            elseif ρ < shrink_threshold
                maximum([0.5*radius, shrink_factor * norm(inv_Dk*sk)])
            end
        else
            radius = 0.5 * radius
        end
        if (radius < min_trust_radius) || (norm(g) < gtol)
            # print gradient convergence
            println("Gradient convergence criterion reached")
            return x, f, g, iter
        end
    end
    println("Maximum number of iterations reached")
    return x, f, g, max_iter
end

"""
    qr_nlss_trust_region(res::Function, jac::Function, x0, lb, ub; kwargs...)

QR-based nonlinear least squares solver with bounded trust region method.
Uses QR factorization for improved numerical stability.

# Arguments
- `res::Function`: Residual function r(x) returning vector of residuals
- `jac::Function`: Jacobian function J(x) returning m×n matrix  
- `x0::Array`: Initial parameter guess
- `lb::Array`: Lower bounds
- `ub::Array`: Upper bounds

# Keyword Arguments
- `initial_radius::Real = 1.0`: Initial trust region radius
- `max_trust_radius::Real = 100.0`: Maximum trust region radius
- `min_trust_radius::Real = 1e-12`: Minimum trust region radius
- `step_threshold::Real = 0.01`: Threshold for accepting steps
- `shrink_threshold::Real = 0.25`: Threshold for shrinking radius
- `expand_threshold::Real = 0.9`: Threshold for expanding radius
- `shrink_factor::Real = 0.25`: Factor for shrinking radius
- `expand_factor::Real = 2.0`: Factor for expanding radius
- `max_iter::Int = 100`: Maximum iterations
- `gtol::Real = 1e-6`: Gradient tolerance
- `ftol::Real = 1e-15`: Function tolerance

# Returns
- Tuple (x, residuals, gradient, iterations)
"""
function qr_nlss_trust_region(
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
    qrls = qr(J, ColumnNorm())
    
    cost = 0.5 * dot(f, f)
    g = compute_gradient(J, f)

    if norm(x0) > 1e-4
        initial_radius = norm(x0)
    end

    
    radius = initial_radius
    
    # Check initial convergence
    if norm(g, Inf) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end

    for iter in 1:max_iter
        # Compute step using QR-based trust region
        #qrls = qr(J)

        m, n = size(J)
        δgn, is_rank_deficient, F = solve_gauss_newton_v3(J, f)
        on_boundary = false
        λ = 0.0

        if m < n
            # --- UNDERDETERMINED CASE (More parameters than residuals) ---
            if norm(δgn) <= radius
                # The minimal-norm step that perfectly fits the model is within the radius.
                # This is the ideal solution. There is no need to make the step longer.
                δ = δgn
                on_boundary = (abs(norm(δgn) - radius) < 1e-9)
            else
                # The smallest "perfect" step is too big. We must find a damped
                # step on the boundary using the standard LM approach.
                λ, δ = find_λ!(radius, J, f, 100)
                on_boundary = true
            end
        else
            # --- SQUARE OR OVERDETERMINED CASE (m >= n) ---
            if norm(δgn) <= radius
                if !is_rank_deficient
                    # Full rank, standard Gauss-Newton step is optimal.
                    δ = δgn
                    on_boundary = (abs(norm(δgn) - radius) < 1e-9)
                else
                    # This is the TRUE "Hard Case" for m >= n systems.
                    on_boundary = true
                    
                    # Extract the null space vector from the SVD we already computed.
                    tol = maximum(size(J)) * eps(F.S[1])
                    effective_rank = count(s -> s > tol, F.S)
                    null_space_idx = effective_rank + 1
                    z = F.V[:, null_space_idx]
                    
                    # Solve for α to extend the step to the boundary.
                    a = dot(z, z)
                    b = 2 * dot(δgn, z)
                    c = dot(δgn, δgn) - radius^2
                    
                    discriminant = b^2 - 4*a*c
                    if discriminant < 0
                        println("Warning: Negative discriminant in hard case. Using GN step.")
                        δ = δgn # Fallback
                    else
                        # Deterministically choose the positive root for α.
                        α = (-b + sqrt(discriminant)) / (2a)
                        δ = δgn + α * z
                    end
                end
            else
                # Standard case: Gauss-Newton step is too long, find damped LM step.
                λ, δ = find_λ!(radius, J, f, 100)
                on_boundary = true
            end
        end

        # δgn, z, hard_case = solve_gauss_newton_v2(J, f)
        # on_boundary = false

        # if norm(δgn) < radius
        #     if !hard_case
        #         δ = δgn
        #         λ = 0.0
        #         on_boundary = false
        #     else #hard case!
        #         # find α: ||δgn||² + 2α(δgn'z) + α²||z||² = Δ²
        #         δgn2 = dot(δgn, δgn)
        #         δgnz = dot(δgn, z)
        #         z2 = dot(z, z)
        #         a = z2
        #         b = 2 * δgnz
        #         c = δgn2 - radius^2
        #         # Solve quadratic equation: aα² + bα + c = 0
        #         discriminant = b^2 - 4a*c
        #         if discriminant < 0
        #             println("No valid step found")
        #             radius *= shrink_factor
        #             continue
        #         end
        #         # α₁ = (-b - sqrt(discriminant)) / (2a)
        #         α₂ = (-b + sqrt(discriminant)) / (2a)
        #         # Calculate the two potential steps
        #         # δ₁ = δgn + α₁ * z
        #         δ₂ = δgn + α₂ * z

        #         # Evaluate the model at each point
        #         # Note: g = J'f, and J'Jδ = J'(Jδ)
        #         # q₁ = -dot(g, δ₁) - 0.5 * dot(J*δ₁, J*δ₁)
        #         # q₂ = -dot(g, δ₂) - 0.5 * dot(J*δ₂, J*δ₂)

        #         # Choose the step that minimizes the model
        #         # if q₁ >= q₂
        #         #     δ = δ₁
        #         # else
        #         #     
        #         # end
        #         δ = δ₂
        #         on_boundary = true
        #     end
            
        # else
        #     # λ = find_λ!(λ₀, radius, J, f, max_iter)
        #     # # println("Iteration: $iter, λ: $λ, radius: $radius")
        #     # δ = qr_trust_region_step(J, qrls, f, radius, lb, ub, x; λ=λ)
        #     λ, δ = find_λ!(radius, J, f, 100)
        #     on_boundary = true
        #     # δ = clamp(δ, lb - x, ub - x)
        # end
        
        if norm(δ) < min_trust_radius
            println("Step size below minimum trust radius")
            return x, f, g, iter
        end
        
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        
        # Compute reduction ratio
        actual_reduction = cost - cost_new
        
        # Predicted reduction using QR factorization
        Jδ = J * δ
        predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        # predicted_reduction = 1/2*norm(Jδ)^2 + λ*norm(δ)^2

        #predicted_reduction =  -(0.5 * sk' * Bk * sk + g' * sk)
        
        if predicted_reduction <= 0
            if actual_reduction > ftol*cost
                x = x_new
                f = f_new
                cost = cost_new
                J = jac(x)
                g = compute_gradient(J, f)
            else
                println("Non-positive predicted reduction, shrinking radius")
                radius *= shrink_factor
            end
            continue
        end
        
        ρ = actual_reduction / predicted_reduction
        
        # Update trust region radius
        if (ρ >= expand_threshold) && on_boundary
            radius = min(max_trust_radius, max(radius, expand_factor * norm(δ)))
        elseif ρ < shrink_threshold
            radius *= shrink_factor
        end
        
        # Accept or reject step
        if ρ >= step_threshold
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = compute_gradient(J, f)
            
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


function lm_fan_lu(
    res::Function, jac::Function,
    x0::Array{T};
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
    M = 1e-8
    μₖ = 1.0
    radius = μₖ * norm(f)
    # Check initial convergence
    if norm(g) < gtol
        println("Initial guess satisfies gradient tolerance")
        return x, f, g, 0
    end
    #iterations
    for iter in 1:max_iter
        # Compute step using QR-based trust region
        δgn = J \ -f
        if norm(δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ!(radius, J, f, 100)
        end
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        # Now solve a new problem based on new x and old J
        δgn_new = J \ -f_new
        radius = μₖ * norm(f_new)
        if norm(δgn_new) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ_new = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ_new = find_λ!(radius, J, f_new, 100)
        end
        x_new_new = x_new + δ_new
        f_new_new = res(x_new_new)
        cost_new_new = 0.5 * dot(f_new_new, f_new_new)
        # Compute reduction ratio
        actual_reduction = cost - cost_new_new
        # Predicted reduction
        Jδ = J * δ
        Jδ_new = J * δ_new
        #predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
        predicted_reduction = -dot(f, Jδ) - 0.5 * dot(Jδ, Jδ) - dot(f_new, Jδ_new) - 0.5 * dot(Jδ_new, Jδ_new)
        if predicted_reduction <= 0 #this potentially means the δ is wrong but we leave some margin
            println("Non-positive predicted reduction, shrinking radius")
            μₖ *= shrink_factor
            radius = μₖ * norm(f)
            continue
        end
        # the reduction ratio
        ρ = actual_reduction / predicted_reduction
        # Update trust region radius
        # Accept or reject step
        if ρ >= step_threshold
            x = x_new
            f = f_new
            cost = cost_new
            J = jac(x)
            g = J' * f
            println("Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius")
            radius = μₖ * norm(f)
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
        if ρ >= expand_threshold
            μₖ *= 4
            radius = μₖ * norm(f)
        elseif ρ < shrink_threshold
            μₖ = max( μₖ/4, M)
            radius = μₖ * norm(f)
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

