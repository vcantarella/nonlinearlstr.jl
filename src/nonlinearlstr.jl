module nonlinearlstr
    include("Affinescale.jl")
    include("trsbox.jl")
    include("tcg.jl")
    include("qr_nlls.jl")
    using LinearAlgebra

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
            H = qr(J'J)
            d_hat = tcgnlss_v2(H, g, radius, dhatl, dhatu, gtol, 1000)
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
            pred_reduction = -(0.5 * sk' * J'J * sk + sk'g)
            actual_reduction = 1/2(norm(f'f) - norm(f_new'f_new))
            pred_reduction = -(0.5 * sk' * H.Q * H.R * sk + g' * sk)
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
    - `regularization::Real = 0.0`: L2 regularization parameter

    # Returns
    - Tuple (x, residuals, gradient, iterations)
    """
    function qr_nlss_bounded_trust_region(
        res::Function, jac::Function,
        x0::Array{T}, lb::Array{T}, ub::Array{T};
        initial_radius::Real = 1.0,
        max_trust_radius::Real = 100.0,
        min_trust_radius::Real = 1e-12,
        step_threshold::Real = 0.01,
        shrink_threshold::Real = 0.25,
        expand_threshold::Real = 0.9,
        shrink_factor::Real = 0.25,
        expand_factor::Real = 2.0,
        max_iter::Int = 100,
        gtol::Real = 1e-6,
        ftol::Real = 1e-15,
        regularization::Real = 0.0,
        ) where T

        # Validate input
        if any(x0 .< lb) || any(x0 .> ub)
            error("Initial guess x0 is not within bounds")
        end

        # Initialize
        x = copy(x0)
        f = res(x)
        J = jac(x)
        qrls = QRLeastSquares(J)
        
        cost = 0.5 * dot(f, f)
        g = compute_gradient(qrls, f)
        
        radius = initial_radius
        
        # Check initial convergence
        if norm(g, Inf) < gtol
            println("Initial guess satisfies gradient tolerance")
            return x, f, g, 0
        end

        for iter in 1:max_iter
            # Update QR factorization
            update!(qrls, J)
            
            # Compute step using QR-based trust region
            try
                δ = qr_trust_region_step(qrls, f, radius, lb, ub, x; λ=regularization)
                
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
                Jδ = qrls.Q * (qrls.R * δ)
                predicted_reduction = -dot(g, δ) - 0.5 * dot(Jδ, Jδ)
                
                if predicted_reduction <= 0
                    println("Non-positive predicted reduction, shrinking radius")
                    radius *= shrink_factor
                    continue
                end
                
                ρ = actual_reduction / predicted_reduction
                
                # Update trust region radius
                if ρ >= expand_threshold
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
                    g = compute_gradient(qrls, f)
                    
                    println("Iteration: $iter, cost: $cost, norm(g): $(norm(g, Inf)), radius: $radius")
                    
                    # Check convergence
                    if norm(g, Inf) < gtol
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
                
            catch e
                println("Error in step computation: $e")
                radius *= shrink_factor
                if radius < min_trust_radius
                    println("Trust region radius below minimum after error")
                    return x, f, g, iter
                end
                continue
            end
        end
        
        println("Maximum number of iterations reached")
        return x, f, g, max_iter
    end
end
