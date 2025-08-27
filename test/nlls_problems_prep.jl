using CUTEst
using NLPModels
using Enlsip
using NLSProblems
using JSOSolvers
using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using LinearAlgebra, Statistics
using BenchmarkTools

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

function find_cutest_nlls_problems(max_vars=50)
    """Find CUTEst nonlinear least squares problems with obj='none' (residual form)"""
    # Find problems with objective="none" (these are NLLS problems)
    nlls_problems = CUTEst.select_sif_problems(objtype="none", max_var=max_vars)
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


function find_nlls_problems(max_vars=50)
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
    n,m = size(jacobian_func(x0))
    # Create objective as 0.5 * ||r||²
    obj_func(x) = 0.5 * dot(residual_func(x), residual_func(x))
    grad_func(x) = jacobian_func(x)' * residual_func(x)

    # Use Gauss-Newton approximation for Hessian
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end
    
    return (
        n=n, m=m, x0=x0, bl=bl, bu=bu,
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        obj_func=obj_func,
        grad_func=grad_func,
        hess_func=hess_func,
        name=prob.meta.name
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
    n,m = size(jacobian_func(x0))
    
    # Create objective as 0.5 * ||r||²
    obj_func(x) = obj(prob, x)
    grad_func(x) = grad(prob, x)
    
    # Use Gauss-Newton approximation for Hessian
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end
    
    return (
        n=n, m=m, x0=x0, bl=bl, bu=bu,
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        obj_func=obj_func,
        grad_func=grad_func,
        hess_func=hess_func,
        name=prob.meta.name
    )
end

function test_solver_on_problem(solver_name, solver_func, prob_data, prob, max_iter=100)
    """Test a single solver on a problem"""
    try
        if solver_name in ["LM-TR", "LM-TR-scaled", "SVD-LM-TR"]
            # Use residual-Jacobian interface
            result = solver_func(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0;
                max_iter=max_iter, gtol=1e-6
            )
            #run one more time for timing:

            t = @benchmark $solver_func(
                $prob_data.residual_func, $prob_data.jacobian_func, 
                $prob_data.x0;
                max_iter=$max_iter, gtol=1e-6)
            elapsed_time = median(t).time
            x_opt, r_opt, g_opt, iterations = result
            final_obj = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, 2) < 1e-6     
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                # because PRIMA counts function evaluations we count an iteration as running the
                #function model n times. I am evaluating time here once because they are the ones that take the longer.
                start_time = time()
                result = PRIMA.newuoa(prob_data.obj_func, prob_data.x0; maxfun=max_iter*prob_data.m)
                elapsed_time = time() - start_time
            else
                lb = prob_data.bl
                ub = prob_data.bu
                start_time = time()
                result = PRIMA.bobyqa(prob_data.obj_func, prob_data.x0; 
                                    xl=lb, xu=ub, maxfun=max_iter*prob_data.m)
                elapsed_time = time() - start_time
            end
            elapsed_time = elapsed_time * 1e9 #from seconds to nanoseconds
            x_opt = result[1]
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.issuccess(result[2])
        elseif contains(solver_name, "NonlinearSolve-") 
            n_res(u, p) = prob_data.residual_func(u)
            nl_jac(u, p) = prob_data.jacobian_func(u)
            nl_func = NonlinearFunction(n_res, jac=nl_jac)
            prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            sol = solve(prob_nl, solver_func(); maxiters=max_iter)
            t = @benchmark solve($prob_nl, $solver_func(); maxiters=$max_iter)
            elapsed_time = median(t).time
            x_opt = sol.u
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)    
        elseif contains(solver_name, "JSO-") 
            stats = solver_func(prob, max_iter=max_iter)
            t = @benchmark $solver_func($prob, max_iter=$max_iter)
            elapsed_time = median(t).time
            x_opt = stats.solution
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order
        elseif contains(solver_name, "LSO-")
            f_func! = (r,x) -> copyto!(r,prob_data.residual_func(x))
            J_func! = (J,x) -> copyto!(J,prob_data.jacobian_func(x))
            res = optimize!(LeastSquaresProblem(x=prob_data.x0, f! = f_func!, 
                                              g! = J_func!, 
                                              output_length=prob_data.m), solver_func)
            t = @benchmark optimize!(LeastSquaresProblem(x=$prob_data.x0, f! = $f_func!,
            g! = $J_func!, output_length=$prob_data.m), $solver_func)
            elapsed_time = median(t).time
            x_opt = res.minimizer
            iterations = res.iterations
            converged = res.converged
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
        # elseif solver_name == "Enlsip"
        #     model = Enlsip.CnlsModel(prob_data.residual_func, prob_data.n, prob_data.m;
        #                             jacobian_residuals=prob_data.jacobian_func, starting_point=prob_data.x0,
        #                             x_low=lb, x_upp=ub)
        #     # Call of the `solve!` function
        #     Enlsip.solve!(model)
        #     status = Enlsip.get_status(model)
        #     println("Algorithm termination status: ", status)
        #     println("Optimal solution: ", Enlsip.solution(model))
        #     println("Optimal objective value: ", Enlsip.sum_sq_residuals(model))

        #     #œresult = Enlsip.enlsip(prob_data.residual_func, prob_data.jacobian_func, prob_data.x0; max_iter=max_iter)
        #     x_opt = Enlsip.solution(model)
        #     final_obj = prob_data.obj_func(x_opt)
        #     g_opt = prob_data.grad_func(x_opt)
        #     iterations = NaN
        #     converged = false
        else
            error("Unknown solver: $solver_name")
        end
        
        return (
            solver = solver_name,
            success = true,
            converged = converged,
            final_objective = final_obj,
            iterations = iterations,
            time = elapsed_time/1e9,  # Convert from nanoseconds to seconds
            final_gradient_norm = norm(g_opt, 2),
            x_opt = x_opt
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
            x_opt = fill(NaN, prob_data.n)
        )
    end
end

function nlls_benchmark(problems, solvers; max_iter=100)
    """Run comprehensive benchmark on CUTEst NLLS problems"""
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
                result = test_solver_on_problem(solver_name, solver_func, prob_data, nlp, max_iter)
                if result.success && result.converged
                    println("✓ obj=$(round(result.final_objective, digits=8)), iters=$(result.iterations)")
                else
                    status = result.success ? "no convergence" : "failed"
                    println("✗ $status")
                end
                result_with_problem = merge(result, (
                    problem = String(prob_name),
                    nvars = prob_data.n,
                    nresiduals = prob_data.m,
                    initial_objective = initial_obj,
                    improvement = initial_obj - result.final_objective,
                ))
                push!(problem_results, result_with_problem)
            end
            append!(results, problem_results)
            # Clean up
            finalize(nlp)
    end
    return results
end
