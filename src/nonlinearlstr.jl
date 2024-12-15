module nonlinearlstr
    include("Affinescale.jl")
    using LinearAlgebra
    using PRIMA

    function bounded_trust_region(func::Function, grad::Function, hess::Function, x0::Array{T}, lb::Array{T}, ub::Array{T}; initial_radius=1,
        max_radius=100, epsilon=1e-9, epsilon_hat = 1e-10, eta = 1e-8, eta_1 = 0.5, eta_2 = 0.9,
        errs = 1e-12, gamma_1 = 0.5, gamma_2 = 1.5, Beta = 0.999, max_iter = 100) where T
        # Check if x0 is within bounds
        if any(x0 .< lb) || any(x0 .> ub)
            error("Initial guess x0 is not within bounds")
        end

        f0 = func(x0)
        g0 = grad(x0)

        Bk = hess(x0)

        #Step 1 termination test:
        if (norm(g0) < epsilon) || (initial_radius < epsilon_hat)
            println("Initial guess is already a minimum gradientwise")
            return x0, f0, g0, 0
        end

        ## initializing the scaling variables:
        ak = zeros(eltype(x0), length(x0))
        bk = zeros(eltype(x0), length(x0))
        Dk = Diagonal(ones(eltype(x0), length(x0)))

        # initializing the x and g vectors
        x = x0
        g = g0
        f = f0
        radius = initial_radius

        for iter in 1:max_iter
            #Step 2: Determine trial step
            ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
            ## Step 2.1 Calculate the scaling vectors and the scaling matrix
            update_vectors_ak_bk!(ak, bk, x, lb, ub)
            update_Dk!(Dk, ak, bk, g, radius, errs)

            ## Step 2.2: Solve the trust region subproblem in the scaled space: d_hat = inv(D)*d
            f_dhat(d_hat) = (Dk*g)'*d_hat + 0.5*d_hat'*(Dk*Bk*Dk)*d_hat
            inv_D = inv(Dk)
            dhatl = inv_D * (lb - x)
            dhatu = inv_D * (ub - x)
            dhat0 = inv_D * -g ./ norm(g)
            dhat0 = dhat0 * radius
            rhobeg = max(norm(dhat0- dhatl), norm(dhatu - dhat0), radius)
            d_hat, info = bobyqa(f_dhat, dhat0, rhobeg = radius, xl = dhatl, xu = dhatu)
            
            if !issuccess(info)
                println("BOBYQA failed to converge")
                println(info)
                error("BOBYQA failed to converge")
            end
            # the update trial solution:
            sk = Beta .* Dk * d_hat

            if norm(sk) < epsilon_hat
                println("Step size convergence criterion reached")
                return x+sk, f, g, iter
            end

            Predk = 0-(g' * sk + 0.5 * sk' * Bk * sk)

            if Predk < epsilon_hat
                println("0 gradient convergence criterion reached")
                return x+sk, f, g, iter
            end
            #Step 3: Test to accept or reject the trial step
            f_new = func(x+sk)
            rho = (f - f_new) / Predk

            if rho >= eta
                x = x + sk
                f = f_new
                g = grad(x)
                Bk = hess(x)
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

            # if (radius < epsilon_hat) || (norm(g) < epsilon)
            #     # print gradient convergence
            #     println("Gradient convergence criterion reached")
            #     return x, f, g, iter
            # end

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

            g = J' * f

            update_Dk!(Dk, ak, bk, g, radius, errs)
            # substep. formulate the subproblem
            inv_D = inv(Dk)

            q(s) = 0.5 * norm(f + J * s)

            f_(d) = 1/2 * norm(f + J * Dk * d)

            dhatl = inv_D * (lb - x)
            dhatu = inv_D * (ub - x)
            dhat0 = inv_D * -g ./ norm(g)
            dhat0 = dhat0 * radius
            rhobeg = max(norm(dhat0- dhatl), norm(dhatu - dhat0), radius)
            d_hat, info = bobyqa(f_, dhat0, rhobeg = radius, xl = dhatl, xu = dhatu)
            
            if !issuccess(info)
                println("BOBYQA failed to converge")
                println(info)
                error("BOBYQA failed to converge")
            end
            # the update trial solution:
            sk = Beta .* Dk * d_hat

            if norm(sk) < epsilon_hat
                println("Step size convergence criterion reached")
                return x+sk, f, g, iter
            end

            fk = norm(f)
            fksk = norm(res(x+sk))
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
