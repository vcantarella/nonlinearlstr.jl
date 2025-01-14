module nonlinearlstr
    include("Affinescale.jl")
    include("trsbox.jl")
    include("tcg.jl")
    using LinearAlgebra
    using PRIMA
    using Roots

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
        if max_trust_radius == nothing
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
            d_hat = tcg(A, b, radius, dhatl, dhatu, gtol, 1000)
            sk = Beta .* Dk * d_hat

            if norm(sk) < gtol
                println("Step size convergence criterion reached")
                return x, f, g, iter
            end

            f_new = func(x+sk)
            actual_reduction = f - f_new
            pred_reduction = -(0.5 * sk' * Bk * sk + g' * sk)
            ρ = actual_reduction / pred_reduction

            # if Predk < gtol
            #     println("0 gradient convergence criterion reached")
            #     return x, f, g, iter
            # end
            #Step 3: Test to accept or reject the trial step

            if ρ >= step_threshold
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


    function nlss_bounded_trust_region(res::Function, jac::Function, x0::Array{T}, lb::Array{T}, ub::Array{T}; initial_radius=1,
        max_radius=100, epsilon=1e-9, epsilon_hat = 1e-10, eta = 1e-8, eta_1 = 0.5, eta_2 = 0.9,
        errs = 1e-12, gamma_1 = 0.5, gamma_2 = 1.5, Beta = 0.999, max_iter = 100) where T
        # Check if x0 is within bounds
        if any(x0 .< lb) || any(x0 .> ub)
            error("Initial guess x0 is not within bounds")
        end

        f0 = res(x0)
        J0 = jac(x0)


        #Step 1 termination test:
        if (norm(J0) < epsilon) || (initial_radius < epsilon_hat)
            println("Initial guess is already a minimum gradientwise")
            return x0, f0, J0, 0
        end

        ## initializing the scaling variables:
        ak = zeros(eltype(x0), length(x0))
        bk = zeros(eltype(x0), length(x0))
        Dk = Diagonal(ones(eltype(x0), length(x0)))

        # initializing the x and g vectors
        x = x0
        J = J0
        f = f0
        radius = initial_radius


        for iter in 1:max_iter
            # Update "Hessina" approx. jacobian
            # Jk = J' * J
            #Step 2: Determine trial step
            ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
            ## Step 2.1 Calculate the scaling vectors and the scaling matrix
            update_vectors_ak_bk!(ak, bk, x, lb, ub)

            g = 2 * J' * f

            update_Dk!(Dk, ak, bk, g, radius, errs)
            # substep. formulate the subproblem
            inv_D = inv(Dk)

            q(s) = f'f + 2f'J*s + s'J'J*s
            J_hat = J * Dk
            f_(d_hat) = f'f + 2f'J_hat*d_hat + d_hat'J_hat'J_hat*d_hat

            dhatl = inv_D * (lb - x)
            dhatu = inv_D * (ub - x)
            dhat0 = inv_D * -g ./ norm(g)
            dhat0 = dhat0 * radius
            
            newton_step = (2J_hat'J_hat)\(-2J_hat'f)
            if norm(newton_step) < radius
                d_hat = newton_step
            else
                steep_descent = (2J_hat'*f)'*(2J_hat'*f)/((2J_hat'f)'*(2J_hat'J_hat)*(2J_hat'*f))*(2J_hat'*f)
                dogleg(τ) = if τ <= 1
                    steep_descent * τ
                else
                    steep_descent + (newton_step - steep_descent)*(τ - 1)
                end
                dogleg_a = newton_step'newton_step -
                 2*newton_step'steep_descent +
                 steep_descent'steep_descent
                dogleg_b = 2*newton_step'steep_descent -
                 2*steep_descent'steep_descent
                dogleg_c = steep_descent'steep_descent-
                radius^2
                f_root(τ) = dogleg_a*(τ-1)^2 + dogleg_b*(τ-1) + dogleg_c
                τ = find_zero(f_root, 1)
                d_hat = dogleg(τ)
            end
            # if !issuccess(info)
            #     println("BOBYQA failed to converge")
            #     println(info)
            #     error("BOBYQA failed to converge")
            # end
            # the update trial solution:
            sk = Dk * d_hat

            # Coleman and Li method for adjusting the step size:

            for i in eachindex(x)
                if x[i] + sk[i] < lb[i] || x[i] + sk[i] > ub[i]
                    sk[i] *= -1
                end
            end
            if any(x + sk .< lb) || any(x + sk .> ub)
                radius = maximum([radius, gamma_2 * norm(inv_D*sk)])
                continue
            end

            if norm(sk) < epsilon_hat
                println("Step size convergence criterion reached")
                return x+sk, f, g, iter
            end

            fk = f'f
            fksk = res(x+sk)'*res(x+sk)
            # fksk = norm(res(x+sk))
            Predk = q(zeros(eltype(sk),size(sk)))-q(sk)
            Areak = fk - fksk

            if Predk < epsilon_hat
                println("0 gradient convergence criterion reached")
                return x+sk, f, g, iter
            end
            #Step 3: Test to accept or reject the trial step
            rho = Areak/Predk

            if rho >= eta
                x = x + sk
                f = res(x)
                J = jac(x)
                println("f: size: $(size(f)), J: size: $(size(J))")
                println("Iteration: $iter, f: $f, norm(g): $(norm(g))")
                println("--------------------------------------------")
            #Step 4: Update the trust region radius
            elseif rho > eta_2
                radius = maximum([radius, gamma_2 * norm(inv_D*sk)])
                if radius > max_radius
                    radius = max_radius
                end
            elseif rho > eta_1
                radius = radius
            elseif rho > eta
                maximum([0.5*radius, gamma_1 * norm(inv_D*sk)])
            else
                radius = 0.5 * radius
            end

            if (radius < epsilon_hat) || (norm(g) < epsilon)
                # print gradient convergence
                println("Gradient convergence criterion reached")
                return x, f, g, iter
            end

        end
        println("Maximum number of iterations reached")
        return x, f, g, max_iter
    end

    function nonlinear_leastsquares(residuals::Function, x0::Array{T}, lb::Array{T}, ub::Array{T};
        xtol=1e-6, ftol=1e-6, gtol=1e-6, max_iter=100, kwargs...) where T
        f(x) = 0.5 * sum(residuals(x).^2)
        g(x) = ForwardDiff.gradient(f, x)
        BoundedTrustRegion(f, g, x0, lb, ub; xtol=xtol, ftol=ftol, gtol=gtol, max_iter=max_iter, kwargs...)
    end

end
