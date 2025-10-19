using LinearAlgebra
include("colemanli.jl")
include("bounded_lm.jl")

function lm_double_trust_region(
    res::Function,
    jac::Function,
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
    m, n = size(J)
    Dk, A = affine_scale_matrix(x, lb, ub, g)
    δgn = [J; √A*Dk] \ [-f; zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter = 1:max_iter
        # Compute step using QR-based trust region
        m, n = size(J)
        if iter > 1
            δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation
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
        Ψ = g'*δ + 1/2*(δ' * (J'J + Ck) * δ)
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
            Dk, A = affine_scale_matrix(x, lb, ub, g)
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
                if (ρᶜ >= expand_threshold)
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
    res::Function,
    jac::Function,
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
    m, n = size(J)
    Dk, A = affine_scale_matrix(x, lb, ub, g)
    δgn = [J; √A*Dk] \ [-f; zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter = 1:max_iter
        # Compute step using QR-based trust region
        m, n = size(J)
        if iter > 1
            δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation
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
        Ψ = g'*δ + 1/2*(δ' * (J'J + Ck) * δ)


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
        Ψ = g'*δ + 1/2*(δ' * (J'J + Ck) * δ)
        ρᶠ = numerator / Ψ # definition of the step improvement
        # Accept or reject step
        if ρᶠ > μ
            x .= x_new
            f .= f_new
            cost = cost_new
            J .= jac(x)
            g .= J' * f
            Dk, A = affine_scale_matrix(x, lb, ub, g)
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
                if (ρᶜ >= expand_threshold)
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


function lm_trust_region_reflective(
    res::Function,
    jac::Function,
    x0::Array{T};
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    μ::Real = 0.005,
    β::Real = 0.1,
    expand_threshold::Real = 0.75,
    shrink_factor::Real = 0.25,
    expand_factor::Real = 2.0,
    max_iter::Int = 100,
    gtol::Real = 1e-6,
    ftol::Real = 1e-15,
    τ::Real = 1e-12,
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
    m, n = size(J)
    Dk, A = affine_scale_matrix(x, lb, ub, g)
    δgn = [J; √A*Dk] \ [-f; zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter = 1:max_iter
        # Compute step using QR-based trust region
        if iter > 1
            δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation
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

        # 2. Calculate alpha_boundary and identify which bounds are hit
        alpha_boundary = Inf
        hits = zeros(Int, n) # To store which bounds are hit

        for i in eachindex(p_k)
            alpha_i = Inf
            hit_i = 0
            if p_k[i] < 0 && lb[i] > -Inf
                alpha_i = (lb[i] - x[i]) / p_k[i]
                hit_i = -1
            elseif p_k[i] > 0 && ub[i] < Inf
                alpha_i = (ub[i] - x[i]) / p_k[i]
                hit_i = 1
            end

            if alpha_i < alpha_boundary
                alpha_boundary = alpha_i
                hits .= 0
                hits[i] = hit_i
            elseif alpha_i == alpha_boundary && hit_i != 0
                hits[i] = hit_i
            end
        end

        # 3. The final step is limited by the boundary and then scaled back
        tau_star = min(1.0, alpha_boundary)
        s_k = (theta * tau_star) * p_k

        # Use s_k (not δ) from here on
        δ = s_k

        Ck = Dk * A * Dk
        Ψ = g'*δ + 1/2*(δ' * (J'J + Ck) * δ)

        # --- REFLECTIVE STEP CALCULATION (Corrected) ---
        if tau_star < 1.0 && any(hits .!= 0)
            # 1. Calculate the step TO the boundary
            s_boundary = tau_star * p_k

            # 2. Define the reflected direction from the boundary point
            p_refl = copy(pₖ)
            p_refl[hits .!= 0] .*= -1.0

            # 3. Perform a line search for the reflected part of the step
            #    The search is for a step `s_part2` along `p_refl`
            #    The total step will be `s_boundary + s_part2`

            # The remaining trust region radius for the second part of the step
            radius_remaining = radius^2 - norm(Dk * s_boundary)^2
            if radius_remaining < 0
                radius_remaining = 0.0
            end
            radius_remaining = sqrt(radius_remaining)

            # Find the intersection with the trust region boundary along p_refl
            norm_Dk_prefl = norm(Dk * p_refl)
            τ_tr = radius_remaining / norm_Dk_prefl

            # Find the intersection with the physical bounds along p_refl
            # starting from the point x + s_boundary
            x_on_boundary = x + s_boundary
            α_boundary_refl = Inf
            for i in eachindex(p_refl)
                if p_refl[i] < 0 && lb[i] > -Inf
                    α_boundary_refl =
                        min(α_boundary_refl, (lb[i] - x_on_boundary[i]) / p_refl[i])
                elseif p_refl[i] > 0 && ub[i] < Inf
                    α_boundary_refl =
                        min(α_boundary_refl, (ub[i] - x_on_boundary[i]) / p_refl[i])
                end
            end
            τ_boundary_refl = Θ * α_boundary_refl

            # The length of the second part of the step is limited by both
            τ_part2 = min(τ_tr, τ_boundary_refl)
            s_part2 = τ_part2 * p_refl

            # The final reflected step is the "dogleg" path
            sᵣ = s_boundary + s_part2

            # Calculate the model value for this new reflected step
            Ψ_refl = g'*sᵣ + 0.5*(sᵣ' * Mₖ * sᵣ)

            # If the reflected step improves the model value, accept it as the new trial step
            if Ψ_refl < Ψ
                println("  -> Accepting reflective step")
                δ = sᵣ
                Ψ = Ψ_refl
            end
        end
        # --- END REFLECTIVE STEP ---
        # 1. Define the scaled gradient direction
        d_grad = -Dk^(-2) * g
        M_k = J'J + Ck

        # 2. Calculate the optimal UNCONSTRAINED step length along d_grad
        # The correct formula is τ = - (g'd) / (d' M d)
        # We must check that the denominator is positive to ensure it's a minimum
        a = dot(d_grad, M_k * d_grad)
        τ_unc = if a > 0
            -dot(g, d_grad) / a
        else
            Inf # It's not a convex path, so the minimum is at the boundary
        end

        # 3. Calculate the step length to the TRUST REGION boundary
        norm_Dk_dg = norm(Dk * d_grad)
        τ_tr = radius / norm_Dk_dg

        # 4. Calculate the step length to the PHYSICAL boundary
        α_boundary_grad = Inf
        for i in eachindex(d_grad)
            if d_grad[i] < 0 && lb[i] > -Inf
                α_boundary_grad = min(α_boundary_grad, (lb[i] - x[i]) / d_grad[i])
            elseif d_grad[i] > 0 && ub[i] < Inf
                α_boundary_grad = min(α_boundary_grad, (ub[i] - x[i]) / d_grad[i])
            end
        end
        # Also apply the step-back factor here
        τ_boundary = Θ * α_boundary_grad

        # 5. The optimal step length is the minimum of all three constraints
        τ_optimal = min(τ_unc, τ_tr, τ_boundary)

        # 6. Calculate the optimal step and the true denominator Ψˢ
        δᵍ = τ_optimal * d_grad
        Ψˢ = dot(g, δᵍ) + 0.5 * dot(δᵍ, M_k * δᵍ)

        # 7. Finally, calculate the true ρᶜ
        # Ensure the denominator isn't zero or positive (model must predict decrease)
        if Ψˢ < Ψ
            δ = δᵍ
            Ψ = Ψˢ
            println("  -> Accepting gradient step")
        end
        # Evaluate new point
        x_new = x + δ
        f_new = res(x_new)
        cost_new = 0.5 * dot(f_new, f_new)
        numerator = cost_new - cost + 1/2*δ'*Ck*δ
        if numerator > 0
            radius *= shrink_factor
            println("step rejected - positive update")
            # Check trust region size
            if radius < min_trust_radius
                println("Trust region radius below minimum")
                return x, f, g, iter
            end
            continue
        end
        Ψ = g'*δ + 1/2*(δ' * (J'J + Ck) * δ) #
        ρᶠ = numerator / Ψ # definition of the step improvement
        # Accept or reject step
        if ρᶠ > μ
            x .= x_new
            f .= f_new
            cost = cost_new
            J .= jac(x)
            g .= J' * f
            Dk, A = affine_scale_matrix(x, lb, ub, g)
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
                if (ρᶜ >= expand_threshold)
                    radius = min(max_trust_radius, expand_factor * radius)
                end
            end
        else
            println("Step rejected, ρᶠ = $ρᶠ")
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


function lm_trust_region_reflective_v2(
    res::Function,
    jac::Function,
    x0::Array{T};
    lb::Array{T} = fill(-Inf, length(x0)),
    ub::Array{T} = fill(Inf, length(x0)),
    initial_radius::Real = 1.0,
    max_trust_radius::Real = 1e12,
    min_trust_radius::Real = 1e-8,
    step_threshold::Real = 0.01,
    expand_threshold::Real = 0.75,
    shrink_factor::Real = 0.25,
    expand_factor::Real = 2.0,
    max_iter::Int = 100,
    gtol::Real = 1e-6,
    ftol::Real = 1e-15,
    τ::Real = 1e-12,
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
    Dk, A, v = affine_scale_matrix(x, lb, ub, g)
    δgn = [J; √A*Dk] \ [-f; zeros(n)]
    initial_radius = norm(Dk*δgn)
    radius = initial_radius
    for iter = 1:max_iter
        # Compute step using QR-based trust region
        if iter > 1
            δgn = [J; √A*Dk] \ [-f; zeros(n)] # avoid extra computation
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

        # 1. Define the trial step and step-back factor
        pₖ = δ
        Θ = 0.995

        # 2. Calculate α_boundary and identify which bounds are hit
        α_boundary = Inf
        hits = zeros(Int, n) # To store which bounds are hit
        for i in eachindex(pₖ)
            αᵢ = Inf
            hit_i = 0
            if pₖ[i] < 0 && lb[i] > -Inf
                αᵢ = (lb[i] - x[i]) / pₖ[i]
                hit_i = -1
            elseif pₖ[i] > 0 && ub[i] < Inf
                αᵢ = (ub[i] - x[i]) / pₖ[i]
                hit_i = 1
            end
            if αᵢ < α_boundary
                α_boundary = αᵢ
                hits .= 0
                hits[i] = hit_i
            elseif αᵢ == α_boundary && hit_i != 0
                hits[i] = hit_i
            end
        end

        Cₖ = Dk * A * Dk
        Mₖ = J'J #+ Cₖ #This is the quadratic element in Coleman and Li

        if α_boundary > 1.0
            δ = pₖ*0.995
            Ψ = g'*δ + 1/2*(δ' * Mₖ * δ)
        else
            # 3. The final step is limited by the boundary and then scaled back
            τ⁺ = min(1.0, α_boundary)
            τ⁺ *= Θ
            # sₖ will be in the line search from 0 to τ⁺
            # a = pₖ'Mₖ*pₖ # a will defined the shape of the parabola (a > 0: convex)
            # if a > 0
            #     #calculate the lowest value:
            #     τᶜ = -g'pₖ / a
            #     if τᶜ < 0
            #         τᶜ = Inf
            #     elseif τᶜ > τ⁺
            #         τᶜ = τ⁺
            #     end
            # else
            #     τᶜ = τ⁺
            # end
            sₖ = τ⁺ * pₖ
            δ = sₖ
            Ψ = g'*δ + 1/2*(δ' * Mₖ * δ)
            # --- REFLECTIVE STEP CALCULATION (Coleman & Li) ---
            # Only consider reflection if the step actually hit a boundary (tau_star < 1.0)
            if τ⁺ < 1.0 && any(hits .!= 0)
                # 1. Calculate the step TO the boundary
                s_boundary = τ⁺ * pₖ

                # 2. Define the reflected direction from the boundary point
                p_refl = copy(pₖ)
                p_refl[hits .!= 0] .*= -1.0

                # 3. Perform a line search for the reflected part of the step
                #    The search is for a step `s_part2` along `p_refl`
                #    The total step will be `s_boundary + s_part2`

                # The remaining trust region radius for the second part of the step
                radius_remaining = radius - norm(Dk * p_refl)
                if radius_remaining < 0
                    radius_remaining = 0.0
                end
                # Find the intersection with the trust region boundary along p_refl
                norm_Dk_prefl = norm(Dk * p_refl)
                τ_tr = radius_remaining / norm_Dk_prefl

                # Find the intersection with the physical bounds along p_refl
                # starting from the point x + s_boundary
                x_on_boundary = x + s_boundary
                α_boundary_refl = Inf
                for i in eachindex(p_refl)
                    if p_refl[i] < 0 && lb[i] > -Inf
                        α_boundary_refl =
                            min(α_boundary_refl, (lb[i] - x_on_boundary[i]) / p_refl[i])
                    elseif p_refl[i] > 0 && ub[i] < Inf
                        α_boundary_refl =
                            min(α_boundary_refl, (ub[i] - x_on_boundary[i]) / p_refl[i])
                    end
                end
                τ_boundary_refl = Θ * α_boundary_refl

                # The length of the second part of the step is limited by both
                τ_part2max = min(τ_tr, τ_boundary_refl)
                # Now we do a line search from 0 to τ_part2:
                a = dot(p_refl, Mₖ * p_refl)
                if a > 0
                    τ_part2 = -dot(g, p_refl) / a
                    if τ_part2 > τ_part2max
                        τ_part2 = τ_part2max
                    else
                        τ_part2 <= 0
                        τ_part2 = 0
                    end
                else
                    τ_part2 = 0 # It's not a convex path, so the minimum is at the boundary
                    # at the boundary or at the full reflective step
                end
                s_part2 = τ_part2 * p_refl

                # The final reflected step is the "dogleg" path
                sᵣ = s_boundary + s_part2

                # Calculate the model value for this new reflected step
                Ψ_refl = g'*sᵣ + 0.5*(sᵣ' * Mₖ * sᵣ)
                # If the reflected step improves the model value, accept it as the new trial step
                if Ψ_refl < Ψ
                    println("  -> Accepting reflective step")
                    δ = sᵣ
                    Ψ = Ψ_refl
                end
            end
            # --- END REFLECTIVE STEP ---
            # 1. Define the scaled gradient direction
            dg = - v .* g # same as: Dk^(-2) * g
            # 2. Calculate the optimal UNCONSTRAINED step length along d_grad
            # The correct formula is τ = - (g'd) / (d' M d)
            # We must check that the denominator is positive to ensure it's a minimum
            a = dot(dg, Mₖ * dg)
            τ_unc = if a > 0
                -dot(g, dg) / a
            else
                Inf # It's not a convex path, so the minimum is at the boundary
            end

            # 3. Calculate the step length to the TRUST REGION boundary
            norm_Dk_dg = norm(Dk * dg)
            τ_tr = Inf #radius / norm_Dk_dg

            # 4. Calculate the step length to the PHYSICAL boundary
            α_boundary_grad = Inf
            for i in eachindex(dg)
                if dg[i] < 0 && lb[i] > -Inf
                    α_boundary_grad = min(α_boundary_grad, (lb[i] - x[i]) / dg[i])
                elseif dg[i] > 0 && ub[i] < Inf
                    α_boundary_grad = min(α_boundary_grad, (ub[i] - x[i]) / dg[i])
                end
            end
            # Also apply the step-back factor here
            τ_boundary = Θ * α_boundary_grad

            # 5. The optimal step length is the minimum of all three constraints
            τ_optimal = min(τ_unc, τ_tr, τ_boundary)

            # 6. Calculate the optimal step and the true denominator Ψˢ
            δᵍ = τ_optimal * dg
            Ψˢ = dot(g, δᵍ) + 0.5 * dot(δᵍ, Mₖ * δᵍ)

            # 7. Finally, calculate the true ρᶜ
            # Ensure the denominator isn't zero or positive (model must predict decrease)
            if Ψˢ < Ψ
                δ = δᵍ
                Ψ = Ψˢ
                println("  -> Accepting gradient step")
            end
        end

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
                Dk, A, v = affine_scale_matrix(x, lb, ub, g)
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
            if (ρᶠ < shrink_factor)
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
