function bounded_step(::ColemanandLiScaling, δ, lb, ub, Dk, A, J, g, x, radius)
    n = length(x)
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
        # v corresponds to components of Dk^(-2).
        # Assuming Dk is Diagonal.
        v = 1 ./ (Dk.diag .^ 2)
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
    return δ, Ψ
end