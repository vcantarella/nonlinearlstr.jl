module nonlinearlstr
    include("Affinescale.jl")
    using LinearAlgebra
    using Prima

    function BoundedTrustRegion(func::Function, grad::Function, x0::Array, lb::Array, ub:Array, initial_radius=1,
        max_radius=100, epsilon=1e-6, epsilon_hat = 1e-10, eta = 1e-8, eta_1 = 0.5, eta_2 = 0.9,
        err = 1e-8, gamma_1 = 0.5, gamma_2 = 1.5, Beta = 0.999, max_iter = 100)
        # Check if x0 is within bounds
        if any(x0 .< lb) || any(x0 .> ub)
            error("Initial guess x0 is not within bounds")
        end

        f0 = func(x0)
        g0 = grad(x0)

        Bk = zeros(eltype(x0), length(x0), length(x0))

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

        for iter in 1:max_iter
            #Step 2: Determine trial step
            ## The problem we approximate by the scaling matrix D and a distance d that we want to solve
            ## Step 2.1 Calculate the scaling vectors and the scaling matrix
            update_vectors_ak_bk!(ak, bk, x, lb, ub)
            tk = calculate_tk(ak, bk, g, initial_radius)
            update_Dk!(Dk, tk, ak, bk, g, initial_radius, err)

            ## Step 2.2: Solve the trust region subproblem in the scaled space: d_hat = inv(D)*d
            f_dhat(d_hat) = (Dk*g)'*d_hat + 0.5*d_hat'*(Dk*Bk*Dk)*d_hat
            inv_D = inv(Dk)
            dhatl = inv_D * (x - lb)
            dhatu = inv_D * (ub - x)
            dhat0 = inv_D * -gk ./ norm(gk) .* initial_radius
            d_hat, info = bobyqa(f_dhat, d_hat0, rhobeg=initial_radius, xl = dhatl, xu = dhatu)
            
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

            Predk = g' * sk + 0.5 * sk' * Bk * sk

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
            end

            #Step 4: Update the trust region radius
            if rho > eta_2
                maximum([radius, gamma_2 * norm(inv_D*sk)])
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

    function nonlinear_leastsquares(residuals::Function, x0::Array, lb::Array, ub::Array,
        xtol=1e-6, ftol=1e-6, gtol=1e-6, max_iter=100)
        f(x) = 0.5 * sum(residuals(x).^2)
        g(x) = ForwardDiff.gradient(f, x)
        BoundedTrustRegion(f, g, x0, lb, ub, epsilon, epsilon_hat, eta, eta_1, eta_2, err, gamma_1, gamma_2, Beta, max_iter)
    end

end
