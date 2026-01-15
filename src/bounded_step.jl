function bounded_step(
    scaling_strategy::ColemanandLiScaling,
    δ,
    lb,
    ub,
    Dk,
    A,
    J,
    g,
    x,
    radius,
    cache,
)
    # Unpack buffers from cache
    hits = cache.hits
    p_refl = cache.p_refl
    dg = cache.dg
    v = cache.v_scaling
    Jv_buff = cache.J_v_buffer
    n = length(x)

    # 1. Define the trial step
    pₖ = δ # Alias
    Θ = 0.995

    # 2. Calculate α_boundary and identify which bounds are hit
    α_boundary = Inf
    fill!(hits, 0)

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
            fill!(hits, 0)
            hits[i] = hit_i
        elseif αᵢ == α_boundary && hit_i != 0
            hits[i] = hit_i
        end
    end

    # Helper function for Quadratic term: v' * M_k * v
    # M_k = J'J + C_k, where C_k = Dk * A * Dk
    # val = ||J*v||^2 + v' * C_k * v
    function compute_quadratic_form(vec_in)
        # 1. J*v part (Allocation free using buffer)
        mul!(Jv_buff, J, vec_in)
        term1 = dot(Jv_buff, Jv_buff)

        # 2. C_k part
        # C_k is diagonal. C_k[i] = Dk[i]^2 * A[i]
        term2 = 0.0
        # Assuming Dk and A are diagonal or vectors
        @inbounds for i = 1:n
            d_val = Dk isa Diagonal ? Dk[i, i] : Dk[i]
            a_val = A isa Diagonal ? A[i, i] : A[i]
            c_val = d_val^2 * a_val
            term2 += vec_in[i]^2 * c_val
        end
        return term1 + term2
    end

    # Helper: u' * M_k * v
    function compute_bilinear_form(u, v)
        # 1. J part: (Ju) ⋅ (Jv)
        # Use existing J_v_buffer for u
        mul!(Jv_buff, J, u)

        # Borrow qtf_buffer for v (it's safe, only used in subproblem solver)
        # Ensure we view it as length m
        m = size(J, 1)
        work_m = view(cache.qtf_buffer, 1:m)
        mul!(work_m, J, v)

        term1 = dot(Jv_buff, work_m)

        # 2. C_k part: u' * C_k * v
        term2 = 0.0
        @inbounds for i = 1:n
            d_val = Dk isa Diagonal ? Dk[i, i] : Dk[i]
            a_val = A isa Diagonal ? A[i, i] : A[i]
            # C_k[i] = D[i]^2 * A[i]
            c_val = d_val^2 * a_val
            term2 += u[i] * v[i] * c_val
        end
        return term1 + term2
    end

    # Initial Model Value
    # Ψ = g'*δ + 0.5 * δ' * Mₖ * δ
    quad_term = compute_quadratic_form(δ)
    Ψ = dot(g, δ) + 0.5 * quad_term

    if α_boundary > 1.0
        # Step is valid, just scale back slightly for safety
        # δ .*= 0.995 # Careful, input δ might be cache.p, don't mutate if used elsewhere or if const
        # Better to return scaled version
        δ = δ * 0.995

        # Recompute Ψ for scaled step
        # Ψ = g'*δ + 0.5 * (0.995^2 * quad_term)
        # Optimization:
        Ψ = dot(g, δ) + 0.5 * (0.995^2 * quad_term)
    else
        # 3. Hit boundary logic
        τ⁺ = min(1.0, α_boundary) * Θ
        sₖ = τ⁺ * pₖ # Allocates vector, usually acceptable or use buffer

        # ... (Line search logic skipped for brevity, standard Trust Region usually takes step to boundary) ...

        δ = sₖ
        quad_term_s = compute_quadratic_form(δ)
        Ψ = dot(g, δ) + 0.5 * quad_term_s

        # --- REFLECTIVE STEP ---
        if τ⁺ < 1.0 && any(hits .!= 0)
            s_boundary = τ⁺ * pₖ

            # Construct p_refl
            copyto!(p_refl, pₖ)
            @inbounds for i = 1:n
                if hits[i] != 0
                    p_refl[i] *= -1.0
                end
            end

            # Radius Logic (Corrected in previous turn)
            norm_s_boundary = norm(Dk * s_boundary)
            radius_remaining = max(0.0, radius - norm_s_boundary)

            norm_Dk_prefl = norm(Dk * p_refl)
            τ_tr = norm_Dk_prefl > 1e-16 ? radius_remaining / norm_Dk_prefl : Inf

            # Boundary Logic for p_refl
            x_on_boundary = x + s_boundary
            α_boundary_refl = Inf
            for i in eachindex(p_refl)
                # ... (Standard boundary check) ...
                if p_refl[i] < 0 && lb[i] > -Inf
                    α_boundary_refl =
                        min(α_boundary_refl, (lb[i] - x_on_boundary[i]) / p_refl[i])
                elseif p_refl[i] > 0 && ub[i] < Inf
                    α_boundary_refl =
                        min(α_boundary_refl, (ub[i] - x_on_boundary[i]) / p_refl[i])
                end
            end
            τ_boundary_refl = Θ * α_boundary_refl

            τ_part2max = min(τ_tr, τ_boundary_refl)

            # Line search
            a_refl = compute_quadratic_form(p_refl)

            slope_term_1 = dot(g, p_refl)
            slope_term_2 = compute_bilinear_form(s_boundary, p_refl)
            g_boundary_dot_p_refl = slope_term_1 + slope_term_2

            if a_refl > 0
                τ_part2 = -g_boundary_dot_p_refl / a_refl #-dot(g, p_refl) / a_refl
                τ_part2 = clamp(τ_part2, 0.0, τ_part2max)
            else
                τ_part2 = 0.0
            end

            s_part2 = τ_part2 * p_refl
            sᵣ = s_boundary + s_part2

            Ψ_refl = dot(g, sᵣ) + 0.5 * compute_quadratic_form(sᵣ)

            if Ψ_refl < Ψ
                println("  -> Accepting reflective step")
                δ = sᵣ
                Ψ = Ψ_refl
            end
        end

        # --- GRADIENT STEP ---
        # v = 1 ./ (Dk.diag .^ 2)
        @inbounds for i = 1:n
            d_val = Dk isa Diagonal ? Dk[i, i] : Dk[i]
            v[i] = 1.0 / (d_val^2)
            dg[i] = -v[i] * g[i]
        end

        # Unconstrained optimal
        a_grad = compute_quadratic_form(dg)
        τ_unc = a_grad > 0 ? -dot(g, dg) / a_grad : Inf

        # Trust region
        norm_Dk_dg = norm(Dk * dg)
        τ_tr = radius / norm_Dk_dg

        # Boundary
        α_boundary_grad = Inf
        for i in eachindex(dg)
            if dg[i] < 0 && lb[i] > -Inf
                α_boundary_grad = min(α_boundary_grad, (lb[i] - x[i]) / dg[i])
            elseif dg[i] > 0 && ub[i] < Inf
                α_boundary_grad = min(α_boundary_grad, (ub[i] - x[i]) / dg[i])
            end
        end
        τ_boundary = Θ * α_boundary_grad

        τ_optimal = min(τ_unc, τ_tr, τ_boundary)

        # Check gradient improvement
        δᵍ = τ_optimal * dg # Allocates
        Ψˢ = dot(g, δᵍ) + 0.5 * compute_quadratic_form(δᵍ)

        if Ψˢ < Ψ
            δ = δᵍ
            Ψ = Ψˢ
            println("  -> Accepting gradient step")
        end
    end

    return δ, Ψ
end
