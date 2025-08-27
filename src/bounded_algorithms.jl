using LinearAlgebra
include("colemanli.jl")
include("bounded_lm.jl")

function lm_double_trust_region(
    res::Function, jac::Function,
    x0::Array{T};
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    μ::Real = 0.25,
    β::Real = 0.1,
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
    Dk, A  = affine_scale_matrix(x, lb, ub, g)
    δgn = [J;√A*Dk] \ [-f;zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter in 1:max_iter
        # Compute step using QR-based trust region
        m, n = size(J)
        if iter > 1
            δgn = [J;√A*Dk] \ [-f;zeros(n)] # avoid extra computation
        end
        if norm(Dk * δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ_scaled_b(radius, J, A, Dk, f, 100)
        end
        #Now we need to ensure strict feasibility (the steps are within bounds)
        # if any(lb .> x .+ δ) || any(ub .< x .+ δ)
        #     println("Step out of bounds, adjusting to stay within bounds")
        #     # Adjust the step to stay within bounds
        #     α = Inf
        #     for i in eachindex(δ)
        #         if lb[i] > -Inf && x[i] + δ[i] < lb[i]
        #             α = min(α, (lb[i] - x[i]) / δ[i])
        #         elseif ub[i] < Inf && x[i] + δ[i] > ub[i]
        #             α = min(α, (ub[i] - x[i]) / δ[i])
        #         end
        #     end
        #     δ .*= α*0.995
        # end

        # 1. Define the trial step and step-back factor
        p_k = δ
        theta = 0.995

        # 2. Calculate alpha_boundary (this must run every time)
        alpha_boundary = Inf
        for i in eachindex(p_k)
            if p_k[i] < 0 && lb[i] > -Inf
                alpha_boundary = min(alpha_boundary, (lb[i] - x[i]) / p_k[i])
            elseif p_k[i] > 0 && ub[i] < Inf
                alpha_boundary = min(alpha_boundary, (ub[i] - x[i]) / p_k[i])
            end
        end

        # 3. The final step is limited by the boundary and then scaled back
        tau_star = min(1.0, alpha_boundary)
        s_k = (theta * tau_star) * p_k

        # Use s_k (not δ) from here on
        δ = s_k 
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        Ck = Dk * A * Dk
        Ψ = g'*δ + 1/2*(δ'* (J'J + Ck) * δ)
        numerator = cost_new - cost + 1/2*δ'*Ck*δ
        ρᶠ = numerator / Ψ # definition of the step improvement
        # in Coleman & Li.
        # a second direction is the reduction along the gradient:
        # δᵍ = -Dk^(-2)*g
        # # we might have to scale this also:
        # τ = δᵍ'*(J'J + Ck)*δᵍ/(g'δᵍ)
        # if any(lb .> x .+ τ*u) || any(ub .< x .+ τ*u)
        #     println("Step out of bounds, adjusting to stay within bounds")
        #     # Adjust the step to stay within bounds
        #     α = Inf
        #     for i in eachindex(δ)
        #         if lb[i] > -Inf && x[i] + τ*u[i] < lb[i]
        #             α = min(α, (lb[i] - x[i]) / (τ*u[i]))
        #         elseif ub[i] < Inf && x[i] + τ*u[i] > ub[i]
        #             α = min(α, (ub[i] - x[i]) / (τ*u[i]))
        #         end
        #     end
        #     τ *= α*0.995
        # end
        # αᵍ = min(αᵍm, τ) # ensure we don't go too small
        # δᵍ = αᵍ * u
        # Ψˢ = g'*δᵍ + 1/2*(δᵍ'* (J'J + Ck) * δᵍ)
        # ρᶜ = 0.2 #Ψ/Ψˢ
        # --- Correct ρᶜ Calculation ---

        # 1. Define the scaled gradient direction
        d_grad = -Dk^(-2) * g
        M_k = J'J + Ck

        # 2. Calculate the optimal UNCONSTRAINED step length along d_grad
        # The correct formula is τ = - (g'd) / (d' M d)
        # We must check that the denominator is positive to ensure it's a minimum
        d_grad_M_d_grad = dot(d_grad, M_k * d_grad)
        tau_unc = if d_grad_M_d_grad > 0
            -dot(g, d_grad) / d_grad_M_d_grad
        else
            Inf # It's not a convex path, so the minimum is at the boundary
        end

        # 3. Calculate the step length to the TRUST REGION boundary
        norm_Dk_d_grad = norm(Dk * d_grad)
        tau_tr = radius / norm_Dk_d_grad

        # 4. Calculate the step length to the PHYSICAL boundary
        alpha_boundary_grad = Inf
        for i in eachindex(d_grad)
            if d_grad[i] < 0 && lb[i] > -Inf
                alpha_boundary_grad = min(alpha_boundary_grad, (lb[i] - x[i]) / d_grad[i])
            elseif d_grad[i] > 0 && ub[i] < Inf
                alpha_boundary_grad = min(alpha_boundary_grad, (ub[i] - x[i]) / d_grad[i])
            end
        end
        # Also apply the step-back factor here
        tau_boundary = 0.995 * alpha_boundary_grad

        # 5. The optimal step length is the minimum of all three constraints
        tau_optimal = min(tau_unc, tau_tr, tau_boundary)

        # 6. Calculate the optimal step and the true denominator Ψˢ
        δᵍ = tau_optimal * d_grad
        Ψˢ = dot(g, δᵍ) + 0.5 * dot(δᵍ, M_k * δᵍ)

        # 7. Finally, calculate the true ρᶜ
        # Ensure the denominator isn't zero or positive (model must predict decrease)
        ρᶜ = Ψˢ < 0 ? Ψ / Ψˢ : Inf 
        
        # Accept or reject step
        if ρᶠ > μ && ρᶜ > β
            x .= x_new
            f .= f_new
            cost = cost_new
            J .= jac(x)
            g .= J' * f
            Dk, A  = affine_scale_matrix(x, lb, ub, g)
            println("Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius")
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
                if  (ρᶜ >= expand_threshold)
                    radius = min(max_trust_radius, expand_factor * radius)
                end
            end
        else
            println("Step rejected, ρᶠ = $ρᶠ, ρᶜ = $ρᶜ")
        end
        if (ρᶠ < μ) || (ρᶜ < β)
            radius *= shrink_factor
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



function lm_interior_and_trust_region(
    res::Function, jac::Function,
    x0::Array{T};
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    μ::Real = 0.1,
    β::Real = 0.1,
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
    Dk, A  = affine_scale_matrix(x, lb, ub, g)
    δgn = [J;√A*Dk] \ [-f;zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter in 1:max_iter
        # Compute step using QR-based trust region
        m, n = size(J)
        if iter > 1
            δgn = [J;√A*Dk] \ [-f;zeros(n)] # avoid extra computation
        end
        if norm(Dk * δgn) <= radius
            # The minimal-norm step that perfectly fits the model is within the radius.
            # This is the ideal solution. There is no need to make the step longer.
            δ = δgn
            λ = 0.0
        else
            # The smallest "perfect" step is too big. We must find a damped
            # step on the boundary using the standard LM approach.
            λ, δ = find_λ_scaled_b(radius, J, A, Dk, f, 100)
        end
        #Now we need to ensure strict feasibility (the steps are within bounds)
        # if any(lb .> x .+ δ) || any(ub .< x .+ δ)
        #     println("Step out of bounds, adjusting to stay within bounds")
        #     # Adjust the step to stay within bounds
        #     α = Inf
        #     for i in eachindex(δ)
        #         if lb[i] > -Inf && x[i] + δ[i] < lb[i]
        #             α = min(α, (lb[i] - x[i]) / δ[i])
        #         elseif ub[i] < Inf && x[i] + δ[i] > ub[i]
        #             α = min(α, (ub[i] - x[i]) / δ[i])
        #         end
        #     end
        #     δ .*= α*0.995
        # end

        # 1. Define the trial step and step-back factor
        p_k = δ
        theta = 0.995

        # 2. Calculate alpha_boundary (this must run every time)
        alpha_boundary = Inf
        for i in eachindex(p_k)
            if p_k[i] < 0 && lb[i] > -Inf
                alpha_boundary = min(alpha_boundary, (lb[i] - x[i]) / p_k[i])
            elseif p_k[i] > 0 && ub[i] < Inf
                alpha_boundary = min(alpha_boundary, (ub[i] - x[i]) / p_k[i])
            end
        end

        # 3. The final step is limited by the boundary and then scaled back
        tau_star = min(1.0, alpha_boundary)
        s_k = (theta * tau_star) * p_k

        # Use s_k (not δ) from here on
        δ = s_k 
        
        Ck = Dk * A * Dk
        Ψ = g'*δ + 1/2*(δ'* (J'J + Ck) * δ)


        # 1. Define the scaled gradient direction
        d_grad = -Dk^(-2) * g
        M_k = J'J + Ck

        # 2. Calculate the optimal UNCONSTRAINED step length along d_grad
        # The correct formula is τ = - (g'd) / (d' M d)
        # We must check that the denominator is positive to ensure it's a minimum
        d_grad_M_d_grad = dot(d_grad, M_k * d_grad)
        tau_unc = if d_grad_M_d_grad > 0
            -dot(g, d_grad) / d_grad_M_d_grad
        else
            Inf # It's not a convex path, so the minimum is at the boundary
        end

        # 3. Calculate the step length to the TRUST REGION boundary
        norm_Dk_d_grad = norm(Dk * d_grad)
        tau_tr = radius / norm_Dk_d_grad

        # 4. Calculate the step length to the PHYSICAL boundary
        alpha_boundary_grad = Inf
        for i in eachindex(d_grad)
            if d_grad[i] < 0 && lb[i] > -Inf
                alpha_boundary_grad = min(alpha_boundary_grad, (lb[i] - x[i]) / d_grad[i])
            elseif d_grad[i] > 0 && ub[i] < Inf
                alpha_boundary_grad = min(alpha_boundary_grad, (ub[i] - x[i]) / d_grad[i])
            end
        end
        # Also apply the step-back factor here
        tau_boundary = 0.995 * alpha_boundary_grad

        # 5. The optimal step length is the minimum of all three constraints
        tau_optimal = min(tau_unc, tau_tr, tau_boundary)

        # 6. Calculate the optimal step and the true denominator Ψˢ
        δᵍ = tau_optimal * d_grad
        Ψˢ = dot(g, δᵍ) + 0.5 * dot(δᵍ, M_k * δᵍ)

        # 7. Finally, calculate the true ρᶜ
        # Ensure the denominator isn't zero or positive (model must predict decrease)
        ρᶜ = Ψˢ < 0 ? Ψ / Ψˢ : Inf

        if ρᶜ < β
            δ = δᵍ
        end
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        numerator = cost_new - cost + 1/2*δ'*Ck*δ
        Ψ = g'*δ + 1/2*(δ'* (J'J + Ck) * δ)
        ρᶠ = numerator / Ψ # definition of the step improvement
        # Accept or reject step
        if ρᶠ > μ
            x .= x_new
            f .= f_new
            cost = cost_new
            J .= jac(x)
            g .= J' * f
            Dk, A  = affine_scale_matrix(x, lb, ub, g)
            println("Iteration: $iter, cost: $cost, norm(g): $(norm(g, 2)), radius: $radius")
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
                if  (ρᶜ >= expand_threshold)
                    radius = min(max_trust_radius, expand_factor * radius)
                end
            end
        else
            println("Step rejected, ρᶠ = $ρᶠ, ρᶜ = $ρᶜ")
        end
        if (ρᶠ < μ)
            radius *= shrink_factor
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