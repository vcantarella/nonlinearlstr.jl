using NLSProblems, NLPModels, LinearAlgebra, Statistics, Random, ForwardDiff
using PRIMA, NonlinearSolve, JSOSolvers
using Pkg, Revise
using DataFrames, CSV, CairoMakie

using nonlinearlstr

function find_nlls_problems(max_vars = 50)
    """Find NLS problems from NLSProblems.jl package"""

    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])

    valid_problems = []

    for prob_name in all_problems
        prob = eval(prob_name)()

        # Filter by size and check if it's a valid NLS problem
        if !unconstrained(prob)
            println("  Problem $prob_name is constrained, skipping")
            finalize(prob)
            continue
        elseif prob.meta.nvar <= max_vars #&& 
            #prob.meta.nequ > 0 &&  # Has residuals
            isa(prob, AbstractNLSModel)

            push!(valid_problems, prob_name)
            finalize(prob)
        else
            finalize(prob)
        end
    end

    println("Found $(length(valid_problems)) valid NLS problems (≤ $max_vars variables)")
    return valid_problems
end

function create_julia_functions(prob)
    """Create Julia function wrappers for NLSProblems problem"""

    # Get problem infor
    #m = prob.meta.nequ  # Number of residuals
    x0 = copy(prob.meta.x0)
    bl = copy(prob.meta.lvar)
    bu = copy(prob.meta.uvar)

    # Define functions using NLPModels interface
    residual_func(x) = residual(prob, x)
    jacobian_func(x) = Matrix(jac_residual(prob, x))
    n, m = size(jacobian_func(x0))

    # Create objective as 0.5 * ||r||²
    obj_func(x) = obj(prob, x)
    grad_func(x) = grad(prob, x)

    # Use Gauss-Newton approximation for Hessian
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end

    return (
        n = n,
        m = m,
        x0 = x0,
        bl = bl,
        bu = bu,
        residual_func = residual_func,
        jacobian_func = jacobian_func,
        obj_func = obj_func,
        grad_func = grad_func,
        hess_func = hess_func,
        name = prob.meta.name,
    )
end

function test_solver_on_problem(solver_name, solver_func, prob_data, prob, max_iter = 100)
    """Test a single solver on a problem"""

    try
        start_time = time()

        if solver_name in ["QR-NLLS", "Standard-NLLS", "QR-NLLS-scaled"]
            # Use residual-Jacobian interface
            result = solver_func(
                prob_data.residual_func,
                prob_data.jacobian_func,
                prob_data.x0,
                prob_data.bl,
                prob_data.bu;
                max_iter = max_iter,
                gtol = 1e-6,
            )
            x_opt, r_opt, g_opt, iterations = result
            final_obj = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, Inf) < 1e-4

        elseif solver_name == "Trust-Region"
            # Use objective-gradient-hessian interface
            result = solver_func(
                prob_data.obj_func,
                prob_data.grad_func,
                prob_data.hess_func,
                prob_data.x0,
                prob_data.bl,
                prob_data.bu;
                max_iter = max_iter,
                gtol = 1e-6,
            )
            x_opt, final_obj, g_opt, iterations = result
            converged = norm(g_opt, Inf) < 1e-4

        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                result = PRIMA.newuoa(prob_data.obj_func, prob_data.x0; maxfun = max_iter)
            else
                # Check if problem has finite bounds
                has_bounds = any(prob_data.bl .> -1e20) || any(prob_data.bu .< 1e20)
                if has_bounds
                    result = PRIMA.bobyqa(
                        prob_data.obj_func,
                        prob_data.x0;
                        xl = prob_data.bl,
                        xu = prob_data.bu,
                        maxfun = max_iter,
                    )
                else
                    # Use large bounds if none specified
                    large_bounds = 1e3
                    result = PRIMA.bobyqa(
                        prob_data.obj_func,
                        prob_data.x0;
                        xl = fill(-large_bounds, prob_data.n),
                        xu = fill(large_bounds, prob_data.n),
                        maxfun = max_iter,
                    )
                end
            end
            x_opt = result[1]
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged =
                PRIMA.reason(result[2]) in [PRIMA.SMALL_TR_RADIUS, PRIMA.FTARGET_ACHIEVED]

        elseif solver_name in ["NL-TrustRegion", "NL-LevenbergMarquardt", "NL-GaussNewton"]
            # Use NonlinearSolve interface for NLLS
            n_res(u, p) = prob_data.residual_func(u)
            nl_jac(u, p) = prob_data.jacobian_func(u)
            nl_func = NonlinearFunction(n_res, jac = nl_jac)

            prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)

            if solver_name == "NL-TrustRegion"
                sol = solve(prob_nl, TrustRegion(); maxiters = max_iter)
            elseif solver_name == "NL-LevenbergMarquardt"
                sol = solve(prob_nl, LevenbergMarquardt(); maxiters = max_iter)
            else  # Gauss-Newton
                sol = solve(prob_nl, GaussNewton(); maxiters = max_iter)
            end

            x_opt = sol.u
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)

        elseif solver_name == "TRON"
            # Use JSOSolvers TRON
            # Create a simple NLPModel wrapper
            stats = tron(prob, max_iter = max_iter)

            x_opt = stats.solution
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order

        elseif solver_name == "TRUNK"
            # Use JSOSolvers TRUNK
            # Create a simple NLPModel wrapper
            stats = trunk(prob, max_iter = max_iter)

            x_opt = stats.solution
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order

        else
            error("Unknown solver: $solver_name")
        end

        elapsed_time = time() - start_time
        bounds_satisfied = all(prob_data.bl .<= x_opt .<= prob_data.bu)

        return (
            solver = solver_name,
            success = true,
            converged = converged,
            final_objective = final_obj,
            iterations = iterations,
            time = elapsed_time,
            bounds_satisfied = bounds_satisfied,
            final_gradient_norm = norm(g_opt, Inf),
            x_opt = x_opt,
        )

    catch e
        println("  $solver_name failed: $e")
        return (
            solver = solver_name,
            success = false,
            converged = false,
            final_objective = Inf,
            iterations = 0,
            time = Inf,
            bounds_satisfied = false,
            final_gradient_norm = Inf,
            x_opt = fill(NaN, prob_data.n),
        )
    end
end
