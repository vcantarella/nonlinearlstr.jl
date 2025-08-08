using CUTEst
using NLPModels
using NLSProblems
using JSOSolvers
using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using LinearAlgebra, Statistics

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
        lb = repeat([-Inf], inner=prob_data.m)
        ub = repeat([Inf], inner=prob_data.m)
        start_time = time()
        if solver_name in ["QR-NLLS", "QR-NLLS-scaled"]
            # Use residual-Jacobian interface
            result = solver_func(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0, lb, ub;
                max_iter=max_iter, gtol=1e-6
            )
            x_opt, r_opt, g_opt, iterations = result
            final_obj = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, 2) < 1e-6     
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                # because PRIMA counts function evaluations we count an iteration as running the
                #function model n times
                result = PRIMA.newuoa(prob_data.obj_func, prob_data.x0; maxfun=max_iter*prob_data.m)
            else
                result = PRIMA.bobyqa(prob_data.obj_func, prob_data.x0; 
                                    xl=lb, xu=ub, maxfun=max_iter*prob_data.m)
            end
            x_opt = result[1]
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.issuccess(result[2])
        elseif contains(solver_name, "NL-") 
            n_res(u, p) = prob_data.residual_func(u)
            nl_jac(u, p) = prob_data.jacobian_func(u)
            nl_func = NonlinearFunction(n_res, jac=nl_jac)
            prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            sol = solve(prob_nl, solver_func(); maxiters=max_iter)   
            x_opt = sol.u
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)    
        elseif (solver_name == "TRON") || (solver_name == "TRUNK")
            stats = solver_func(prob, max_iter=max_iter)
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
            x_opt = res.minimizer
            iterations = res.iterations
            converged = res.converged
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
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


function analyze_cutest_results(results)
    """Analyze and visualize CUTEst benchmark results"""
    
    # Convert to DataFrame
    df = DataFrame(results)
    
    # Filter runs that completed without error
    completed = filter(r -> r.success && isfinite(r.final_objective), df)
    
    # For each problem, find the best (minimum) objective across all solvers
    problem_best_objectives = Dict{String, Float64}()
    for prob in unique(completed.problem)
        prob_results = filter(r -> r.problem == prob, completed)
        if nrow(prob_results) > 0
            best_obj = minimum(prob_results.final_objective)
            problem_best_objectives[prob] = best_obj
        end
    end
    
    # Define success criterion: final objective within 1e-6 relative tolerance of best
    # or within 1e-8 absolute tolerance for very small objectives
    function is_successful(final_obj, best_obj)
        if abs(best_obj) < 1e-8
            return abs(final_obj - best_obj) < 1e-8
        else
            return abs(final_obj - best_obj) / abs(best_obj) < 1e-6
        end
    end
    
    # Filter for truly successful runs (found approximately the best solution)
    successful = filter(completed) do r
        if haskey(problem_best_objectives, r.problem)
            best_obj = problem_best_objectives[r.problem]
            return is_successful(r.final_objective, best_obj)
        else
            return false
        end
    end
    
    println("\n" * "="^80)
    println("CUTEST NLLS BENCHMARK RESULTS")
    println("="^80)
    
    # Overall statistics
    total_problems = length(unique(df.problem))
    println("Total problems tested: $total_problems")
    println("Success criterion: final objective within 1e-6 relative tolerance of best found")
    
    for solver in unique(df.solver)
        solver_completed = filter(r -> r.solver == solver, completed)
        solver_successful = filter(r -> r.solver == solver, successful)
        
        completion_rate = nrow(solver_completed) / total_problems * 100
        success_rate = nrow(solver_successful) / total_problems * 100
        
        if nrow(solver_successful) > 0
            avg_time = mean(solver_successful.time)
            avg_iters = mean(solver_successful.iterations)
            
            # Count how many times this solver found the best solution
            best_count = 0
            for prob in unique(successful.problem)
                prob_successful = filter(r -> r.problem == prob, successful)
                solver_result = filter(r -> r.solver == solver && r.problem == prob, prob_successful)
                if nrow(solver_result) > 0
                    best_count += 1
                end
            end
            
            println("\n$solver:")
            println("  Completion rate: $(round(completion_rate, digits=1))%")
            println("  Success rate: $(round(success_rate, digits=1))%")
            println("  Avg time (successful): $(round(avg_time, digits=4))s")
            println("  Avg iterations (successful): $(round(avg_iters, digits=1))")
            println("  Problems where found best solution: $best_count")
        else
            println("\n$solver:")
            println("  Completion rate: $(round(completion_rate, digits=1))%")
            println("  Success rate: 0.0%")
            println("  No successful runs")
        end
    end
    
    # Problem-by-problem comparison
    println("\n" * "-"^80)
    println("PROBLEM-BY-PROBLEM RESULTS")
    println("-"^80)
    
    for prob in sort(unique(successful.problem))
        prob_successful = filter(r -> r.problem == prob, successful)
        if nrow(prob_successful) > 0
            best_obj = problem_best_objectives[prob]
            fastest = minimum(prob_successful.time)
            
            # Get first row info
            prob_info = prob_successful[1, :]
            bounds_str = prob_info.has_bounds ? "bounded" : "unconstrained"
            
            println("\n$prob ($(prob_info.nvars) vars, $(prob_info.nresiduals) res, $bounds_str):")
            println("  Best objective found: $(round(best_obj, digits=10))")
            
            # Sort by time for successful runs
            sorted_results = sort(prob_successful, :time)
            for row in eachrow(sorted_results)
                is_fastest = row.time ≈ fastest
                marker = is_fastest ? "⚡" : "  "
                
                println("  $(rpad(row.solver, 20)): obj=$(round(row.final_objective, digits=10)), " *
                       "time=$(round(row.time, digits=4))s, iters=$(row.iterations) $marker")
            end
        end
    end
    
    # Solver head-to-head comparison (only among successful runs)
    println("\n" * "-"^80)
    println("SOLVER SUCCESS COMPARISON")
    println("-"^80)
    
    solvers = sort(unique(df.solver))
    
    # Count problems where each solver succeeded
    success_matrix = zeros(Int, length(solvers))
    
    for (i, solver) in enumerate(solvers)
        solver_successful = filter(r -> r.solver == solver, successful)
        success_matrix[i] = nrow(solver_successful)
    end
    
    # Print success counts
    println("Problems successfully solved by each solver:")
    for (i, solver) in enumerate(solvers)
        percentage = success_matrix[i] / total_problems * 100
        println("$(rpad(solver, 25)): $(success_matrix[i])/$total_problems ($(round(percentage, digits=1))%)")
    end
    
    # Speed comparison among successful runs
    println("\n" * "-"^80)
    println("SPEED COMPARISON (Successful Runs Only)")
    println("-"^80)
    
    for solver in solvers
        solver_successful = filter(r -> r.solver == solver, successful)
        if nrow(solver_successful) > 0
            times = solver_successful.time
            println("$(rpad(solver, 25)): median=$(round(median(times), digits=4))s, " *
                   "mean=$(round(mean(times), digits=4))s, " *
                   "min=$(round(minimum(times), digits=4))s")
        end
    end
    
    # Save results with success flag
    df_with_success = copy(df)
    df_with_success.truly_successful = map(eachrow(df_with_success)) do row
        if haskey(problem_best_objectives, row.problem)
            best_obj = problem_best_objectives[row.problem]
            return row.success && is_successful(row.final_objective, best_obj)
        else
            return false
        end
    end
    
    CSV.write("cutest_nlls_results.csv", df_with_success)
    println("\nResults saved to cutest_nlls_results.csv")
    
    return df_with_success
end

function plot_cutest_performance(df)
    """Create performance plots showing fraction of problems solved vs time (only truly successful runs)"""
    
    # Filter for truly successful runs only
    successful = filter(r -> r.truly_successful, df)
    
    if nrow(successful) == 0
        println("No truly successful results to plot")
        return nothing
    end
    
    fig = Figure(size=(1200, 800))
    
    # Plot 1: Fraction solved vs time
    ax1 = Axis(fig[1, 1], 
               xlabel="Time (seconds, log scale)", 
               ylabel="Fraction of Problems Solved",
               title="Performance Profile - Fraction Successfully Solved vs Time",
               xscale=log10)
    
    # Plot 2: Success rate comparison
    ax2 = Axis(fig[1, 2], 
               xlabel="Solver", 
               ylabel="Success Rate (%)",
               title="Success Rate by Solver")
    
    solvers = sort(unique(df.solver))
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :cyan]
    
    total_problems = length(unique(df.problem))
    println("Total problems: $total_problems")
    println("Success criterion: within 1e-6 relative tolerance of best objective")
    
    success_rates = Float64[]
    solver_names = String[]
    
    for (i, solver) in enumerate(solvers)
        solver_successful = filter(r -> r.solver == solver, successful)
        
        n_successful = nrow(solver_successful)
        success_rate = n_successful / total_problems * 100
        
        push!(success_rates, success_rate)
        push!(solver_names, solver)
        
        if n_successful == 0
            println("$solver: solved 0/$total_problems problems (0.0%)")
            continue
        end
        
        # Get all times for this solver (one per problem)
        times = Float64[]
        for prob in unique(solver_successful.problem)
            prob_result = filter(r -> r.problem == prob, solver_successful)
            if nrow(prob_result) > 0
                push!(times, prob_result[1, :time])
            end
        end
        
        if isempty(times)
            continue
        end
        
        # Sort times
        sorted_times = sort(times)
        n_solved = length(sorted_times)
        
        # Calculate maximum fraction this solver can achieve
        max_fraction = n_solved / total_problems
        
        # Create cumulative fraction array
        fractions = (1:n_solved) / total_problems
        
        # Add point at start (0 problems solved at t=0)
        min_time = minimum(sorted_times)
        plot_times = vcat([min_time * 0.1], sorted_times)
        plot_fractions = vcat([0.0], fractions)
        
        # Fix: Use modulo to cycle through colors if there are more solvers than colors
        color = colors[((i-1) % length(colors)) + 1]
        
        # Plot step function
        lines!(ax1, plot_times, plot_fractions, 
              color=color, linewidth=2, 
              label="$solver ($(round(max_fraction*100, digits=1))%)")
        
        println("$solver: solved $n_solved/$total_problems problems ($(round(max_fraction*100, digits=1))%)")
    end
    
    # Add horizontal line at 100%
    hlines!(ax1, [1.0], color=:black, linestyle=:dash, alpha=0.5)
    
    # Set y-axis limits
    ylims!(ax1, 0, 1.05)
    axislegend!(ax1, position=:rb)
    
    # Bar chart of success rates - Fix: Make sure we don't exceed color array bounds
    n_solvers = length(solvers)
    bar_colors = [colors[((i-1) % length(colors)) + 1] for i in 1:n_solvers]
    
    barplot!(ax2, 1:n_solvers, success_rates, 
            color=bar_colors, alpha=0.7)
    
    # Fix: Make sure we don't try to slice more solver names than we have
    truncated_names = [s[1:min(10,length(s))] for s in solver_names]
    ax2.xticks = (1:n_solvers, truncated_names)
    ax2.xticklabelrotation = π/4
    ylims!(ax2, 0, max(maximum(success_rates) * 1.1, 10))
    
    # Add percentage labels on bars
    for (i, rate) in enumerate(success_rates)
        if rate > 0
            text!(ax2, i, rate + 1, text="$(round(rate, digits=1))%", 
                  align=(:center, :bottom), fontsize=10)
        end
    end
    
    save("cutest_nlls_performance.png", fig)
    display(fig)
    return fig
end

# # Run the benchmark
# println("Starting CUTEst NLLS benchmark...")
# results = cutest_nlls_benchmark(15, 200)  # Test 15 problems, max 200 iterations

# # Analyze results
# if !isempty(results)
#     df_results = analyze_cutest_results(results)
    
#     # Create performance plots
#     fig = plot_cutest_performance(df_results)
    
#     println("\nCUTEst benchmark complete!")
#     println("Results saved to cutest_nlls_results.csv")
#     println("Performance plots saved to cutest_nlls_performance.png")
# else
#     println("No results to analyze")
# end