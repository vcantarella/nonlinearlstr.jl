using CUTEst
using NLPModels
using NLSProblems
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using Test
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using LinearAlgebra, Statistics
using NLLSsolver
using StaticArrays
using BenchmarkTools
using CondaPkg
using PythonCall
using StaticArrays
using Static
scipy = pyimport("scipy")

using nonlinearlstr

# Generic wrapper for prob_data residual
# Parametric wrapper encodes sizes at the type level so NLLSsolver can use
# static sizes while we still carry a runtime `prob_data` instance.
struct ProbDataResidual{NB,M} <: NLLSsolver.AbstractResidual
    prob_data::Any
end

# Construct a properly-parameterized wrapper from runtime `prob_data`.
# NB is the number of variable blocks the residual depends on (we use 1).
ProbDataResidual(prob_data) = ProbDataResidual{1,prob_data.n}(prob_data)

Base.eltype(::ProbDataResidual) = Float64
NLLSsolver.ndeps(::ProbDataResidual{NB,M}) where {NB,M} = static(NB) # number of variable blocks
NLLSsolver.nres(::ProbDataResidual{NB,M}) where {NB,M} = static(M)   # residual length
NLLSsolver.varindices(::ProbDataResidual{NB,M}) where {NB,M} = SVector{NB}(1:NB)

function NLLSsolver.getvars(::ProbDataResidual{NB,M}, vars::Vector) where {NB,M}
    # Return the variable blocks this residual depends on as a tuple
    return (vars[1],)
end

function NLLSsolver.computeresidual(res::ProbDataResidual{NB,M}, vars...) where {NB,M}
    # `vars` may be passed as a tuple of variable blocks or explicit args
    vb = length(vars) == 1 ? vars[1] : vars
    x = collect(vb)
    return SVector{M}(res.prob_data.residual_func(x))
end

# Provide an analytic residual+jacobian implementation so the solver uses it
# rather than attempting autodiff through potentially non-Dual-friendly code.
function NLLSsolver.computeresjacstatic(
    varflags::StaticInt{NB},
    res::ProbDataResidual{NB,M},
    vars,
) where {NB,M}
    vb = isa(vars, Tuple) ? vars[1] : vars
    x = collect(vb)
    r = SVector{M}(res.prob_data.residual_func(x))
    J = res.prob_data.jacobian_func(x)
    return r, J
end

function find_cutest_nlls_problems(max_vars = 50)
    """Find CUTEst nonlinear least squares problems with obj='none' (residual form)"""
    # Find problems with objective="none" (these are NLLS problems)
    nlls_problems = CUTEst.select_sif_problems(objtype = "none", max_var = max_vars)
    # Test which ones actually have residuals
    valid_problems = []
    for prob_name in nlls_problems
        nlp = CUTEstModel(prob_name)
        # Check if it has constraints (residuals for NLLS)
        if nlp.meta.ncon > 0 && nlp.meta.nvar <= max_vars
            push!(valid_problems, prob_name)
        end
        finalize(nlp)
    end
    println("Found $(length(valid_problems)) valid NLLS problems")
    return valid_problems
end


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

function find_bounded_problems(max_vars = Inf)
    """Find bounded NLS problems from NLSProblems.jl package"""
    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])

    valid_problems = []

    for prob_name in all_problems
        prob = eval(prob_name)()

        # Filter by size and check if it's a valid NLS problem
        if !bound_constrained(prob)
            # println("  Problem $prob_name is not bound constrained, skipping")
            finalize(prob)
            continue
        elseif prob.meta.nvar <= max_vars
            isa(prob, AbstractNLSModel)

            push!(valid_problems, prob_name)
            finalize(prob)
        else
            finalize(prob)
        end
    end

    println(
        "Found $(length(valid_problems)) valid bounded NLS problems (≤ $max_vars variables)",
    )
    return valid_problems
end

function create_cutest_functions(nlp)
    """Create Julia function wrappers for CUTEst NLLS problem"""
    # Get problem info
    #n = nlp.meta.nvar
    #m = nlp.meta.ncon  # Number of residuals (constraints in NLLS formulation)
    x0 = copy(nlp.meta.x0)
    bl = copy(nlp.meta.lvar)
    bu = copy(nlp.meta.uvar)
    # For CUTEst NLLS problems with objtype="none", the residuals are the constraints
    residual_func(x) = NLPModels.cons(nlp, x)
    jacobian_func(x) = Matrix(NLPModels.jac(nlp, x))
    n, m = size(jacobian_func(x0))
    # Create objective as 0.5 * ||r||²
    obj_func(x) = 0.5 * dot(residual_func(x), residual_func(x))
    grad_func(x) = jacobian_func(x)' * residual_func(x)

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
        initial_cost = obj_func(x0),
        residual_func = residual_func,
        jacobian_func = jacobian_func,
        obj_func = obj_func,
        grad_func = grad_func,
        hess_func = hess_func,
        problem = nlp.meta.name,
    )
end


function create_nls_functions(prob)
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
        initial_cost = obj_func(x0),
        residual_func = residual_func,
        jacobian_func = jacobian_func,
        obj_func = obj_func,
        grad_func = grad_func,
        hess_func = hess_func,
        problem = prob.meta.name,
    )
end

function test_solver_on_problem(solver_name, solver_func, prob_data, prob, max_iter = 100)
    """Test a single solver on a problem"""
    try
        if solver_name in [
            "LM-QR",
            "LM-QR-scaled",
            "LM-SVD",
            "LM-QR-Recursive",
            "LM-QR-Recursive-Scaled",
            "LM-QR-Scaled",
        ]
            # Use residual-Jacobian interface
            if contains(solver_name, "scaled") || contains(solver_name, "Scaled")
                scaling_strategy = nonlinearlstr.JacobianScaling()
            else
                scaling_strategy = nonlinearlstr.NoScaling()
            end
            if contains(solver_name, "Recursive")
                subproblem_strategy = nonlinearlstr.QRrecursiveSolve()
            elseif contains(solver_name, "QR")
                subproblem_strategy = nonlinearlstr.QRSolve()
            else # contains(solver_name, "SVD")
                subproblem_strategy = nonlinearlstr.SVDSolve()
            end

            result = solver_func(
                prob_data.residual_func,
                prob_data.jacobian_func,
                prob_data.x0,
                subproblem_strategy,
                scaling_strategy;
                max_iter = max_iter,
                gtol = 1e-6,
            )
            #run one more time for timing:

            t = @elapsed solver_func(
                prob_data.residual_func,
                prob_data.jacobian_func,
                prob_data.x0,
                subproblem_strategy,
                scaling_strategy;
                max_iter = max_iter,
                gtol = 1e-6,
            )

            x_opt, r_opt, g_opt, iterations = result
            final_cost = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, 2) < 1e-6
        elseif solver_name in ["TRF", "TRF-scaled"]
            scaling_strategy = nonlinearlstr.ColemanandLiScaling()

            result = solver_func(
                prob_data.residual_func,
                prob_data.jacobian_func,
                prob_data.x0;
                lb = prob_data.bl,
                ub = prob_data.bu,
                max_iter = max_iter,
                gtol = 1e-6,
            )
            #run one more time for timing:

            t = @elapsed solver_func(
                prob_data.residual_func,
                prob_data.jacobian_func,
                prob_data.x0;
                lb = prob_data.bl,
                ub = prob_data.bu,
                max_iter = max_iter,
                gtol = 1e-6,
            )

            x_opt, r_opt, g_opt, iterations = result
            final_cost = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, 2) < 1e-6
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                # because PRIMA counts function evaluations we count an iteration as running the
                #function model n times. I am evaluating time here once because they are the ones that take the longer.
                start_time = time()
                result = PRIMA.newuoa(
                    prob_data.obj_func,
                    prob_data.x0;
                    maxfun = max_iter*prob_data.m,
                )
                t = time() - start_time
            else
                lb = prob_data.bl
                ub = prob_data.bu
                start_time = time()
                result = PRIMA.bobyqa(
                    prob_data.obj_func,
                    prob_data.x0;
                    xl = lb,
                    xu = ub,
                    maxfun = max_iter*prob_data.m,
                )
                t = time() - start_time
            end
            x_opt = result[1]
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.issuccess(result[2])
        elseif contains(solver_name, "NonlinearSolve-")
            n_res(u, p) = prob_data.residual_func(u)
            nl_jac(u, p) = prob_data.jacobian_func(u)
            nl_func = NonlinearFunction(n_res, jac = nl_jac)
            prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            sol = solve(prob_nl, solver_func(); maxiters = max_iter)
            t = @elapsed solve(prob_nl, solver_func(); maxiters = max_iter)
            x_opt = sol.u
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)
            # elseif contains(solver_name, "MINPACK-")
            #     n_res(u, p) = residual(nlp, u)
            #     nl_jac(u, p) = Matrix(jac_residual(prob, u))
            #     nl_func = NonlinearFunction(n_res, jac = nl_jac)
            #     prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            #     sol = solve(prob_nl, solver_func(); maxiters = max_iter)
            #     t = @elapsed solve(prob_nl, solver_func(); maxiters = max_iter)
            #     x_opt = sol.u
            #     final_cost = prob_data.obj_func(x_opt)
            #     g_opt = prob_data.grad_func(x_opt)
            #     iterations = sol.stats.nsteps
            #     converged = SciMLBase.successful_retcode(sol)
        elseif contains(solver_name, "JSO-")
            stats = solver_func(prob, max_iter = max_iter)
            t = @elapsed solver_func(prob, max_iter = max_iter)
            x_opt = stats.solution
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order
        elseif contains(solver_name, "LSO-")
            f_func! = (r, x) -> copyto!(r, prob_data.residual_func(x))
            J_func! = (J, x) -> copyto!(J, prob_data.jacobian_func(x))
            res = LeastSquaresOptim.optimize!(
                LeastSquaresProblem(
                    x = prob_data.x0,
                    f! = f_func!,
                    g! = J_func!,
                    output_length = prob_data.n,
                ),
                solver_func,
                lower = prob_data.bl,
                upper = prob_data.bu,
            )
            t = @elapsed LeastSquaresOptim.optimize!(
                LeastSquaresProblem(
                    x = prob_data.x0,
                    f! = f_func!,
                    g! = J_func!,
                    output_length = prob_data.n,
                ),
                solver_func,
                lower = prob_data.bl,
                upper = prob_data.bu,
            )
            x_opt = res.minimizer
            iterations = res.iterations
            converged = res.converged
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
        elseif solver_name == "Scipy-LeastSquares"
            pyresult = scipy.optimize.least_squares(
                prob_data.residual_func,
                prob_data.x0,
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                xtol = 1e-8,
                gtol = 1e-6,
                max_nfev = 1000,
                verbose = 0,
            )
            t = @elapsed scipy.optimize.least_squares(
                prob_data.residual_func,
                prob_data.x0,
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                xtol = 1e-8,
                gtol = 1e-6,
                max_nfev = 1000,
                verbose = 0,
            )
            x_opt = pyconvert(Vector{Float64}, pyresult["x"])
            final_cost =
                0.5 * dot(prob_data.residual_func(x_opt), prob_data.residual_func(x_opt))
            converged = pyconvert(Bool, pyresult["success"])
            g_opt = prob_data.grad_func(x_opt)
            iterations = pyconvert(Int, pyresult["njev"])
            pyresult = nothing
            GC.gc()
        elseif solver_name == "Scipy-LSMR"
            pyresult = scipy.optimize.least_squares(
                prob_data.residual_func,
                prob_data.x0,
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                tr_solver = "lsmr",
                xtol = 1e-8,
                gtol = 1e-6,
                max_nfev = 1000,
                verbose = 0,
            )
            t = @elapsed scipy.optimize.least_squares(
                prob_data.residual_func,
                prob_data.x0,
                jac = prob_data.jacobian_func,
                bounds = (prob_data.bl, prob_data.bu),
                xtol = 1e-8,
                gtol = 1e-6,
                max_nfev = 1000,
                verbose = 0,
            )
            x_opt = pyconvert(Vector{Float64}, pyresult["x"])
            final_cost =
                0.5 * dot(prob_data.residual_func(x_opt), prob_data.residual_func(x_opt))
            converged = pyconvert(Bool, pyresult["success"])
            g_opt = prob_data.grad_func(x_opt)
            iterations = pyconvert(Int, pyresult["njev"])
            pyresult = nothing
            GC.gc()
        elseif contains(solver_name, "NLLSsolver-")
            # Build and populate the NLLSsolver problem in one block to avoid
            # REPL/display trying to summarize an empty problem (which triggers
            # reductions over empty collections inside NLLSsolver.show).
            res_obj = ProbDataResidual(prob_data)
            # create concrete types from runtime sizes
            var_type = typeof(NLLSsolver.EuclideanVector(prob_data.x0...))
            res_type = ProbDataResidual{1,prob_data.n}
            problem = let p = NLLSsolver.NLLSProblem(var_type, res_type)
                NLLSsolver.addvariable!(p, NLLSsolver.EuclideanVector(prob_data.x0...))
                NLLSsolver.addcost!(p, res_obj)
                p
            end
            options = NLLSsolver.NLLSOptions(
                reldcost = 1e-11,
                iterator = solver_func,
                maxiters = max_iter,
            )
            result = NLLSsolver.optimize!(problem, options)
            t = @elapsed NLLSsolver.optimize!(problem, options)
            x_opt = collect(problem.variables[1])
            final_cost = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result.niterations
            converged = result.termination > 0
        else
            error("Unknown solver: solver_name")
        end

        bounds_satisfied = all(prob_data.bl .<= x_opt .<= prob_data.bu)

        return (
            solver = solver_name,
            success = true,
            converged = converged,
            final_cost = final_cost,
            iterations = iterations,
            time = t,
            bounds_satisfied = bounds_satisfied,
            final_gradient_norm = norm(g_opt, 2),
            x_opt = x_opt,
        )
    catch e
        println("  $solver_name failed: $e")
        return (
            solver = solver_name,
            success = false,
            converged = false,
            final_cost = Inf,
            iterations = 0,
            time = Inf,
            bounds_satisfied = false,
            final_gradient_norm = Inf,
            x_opt = fill(NaN, prob_data.n),
        )
    end
end


function nlls_benchmark(problems, solvers; max_iter = 100)
    """Run comprehensive elapsed on NLLS problems"""
    # Define solvers to test
    results = []
    max_problems = length(problems)
    for (i, prob_name) in enumerate(problems)
        println("\n" * "="^60)
        println("Problem $i/$max_problems: $prob_name")
        # Create problem instance
        local nlp
        local prob_data
        if isa(prob_name, String)
            nlp = CUTEstModel(prob_name)
            prob_data = create_cutest_functions(nlp)
        else
            nlp = eval(prob_name)()
            prob_data = create_nls_functions(nlp)
        end
        # Create Julia functions
        println("  Variables: $(prob_data.n)")
        println("  Residuals: $(prob_data.m)")
        initial_obj = prob_data.obj_func(prob_data.x0)
        println("  Initial objective: $initial_obj")
        # Test each solver
        problem_results = []
        for (solver_name, solver_func) in solvers
            print("    Testing $solver_name... ")
            result =
                test_solver_on_problem(solver_name, solver_func, prob_data, nlp, max_iter)
            if result.success && result.converged
                println(
                    "✓ obj=$(round(result.final_cost, digits=8)), iters=$(result.iterations)",
                )
            else
                status = result.success ? "no convergence" : "failed"
                println("✗ $status")
            end
            result_with_problem = merge(
                result,
                (
                    problem = String(prob_name),
                    nvars = prob_data.n,
                    nresiduals = prob_data.m,
                    initial_objective = initial_obj,
                    improvement = initial_obj - result.final_cost,
                ),
            )
            push!(problem_results, result_with_problem)
        end
        append!(results, problem_results)
        # Clean up
        finalize(nlp)
    end
    return results
end
