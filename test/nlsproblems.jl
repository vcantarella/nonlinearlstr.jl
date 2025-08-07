using NLSProblems, NLPModels, LinearAlgebra, Statistics, Random, ForwardDiff
using PRIMA, NonlinearSolve, JSOSolvers
using Pkg, Revise
using DataFrames, CSV, CairoMakie

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

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
        start_time = time()
        
        if solver_name in ["QR-NLLS", "Standard-NLLS", "QR-NLLS-scaled"]
            # Use residual-Jacobian interface
            result = solver_func(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0, prob_data.bl, prob_data.bu;
                max_iter=max_iter, gtol=1e-6
            )
            x_opt, r_opt, g_opt, iterations = result
            final_obj = 0.5 * dot(r_opt, r_opt)
            converged = norm(g_opt, Inf) < 1e-4
            
        elseif solver_name == "Trust-Region"
            # Use objective-gradient-hessian interface
            result = solver_func(
                prob_data.obj_func, prob_data.grad_func, prob_data.hess_func,
                prob_data.x0, prob_data.bl, prob_data.bu;
                max_iter=max_iter, gtol=1e-6
            )
            x_opt, final_obj, g_opt, iterations = result
            converged = norm(g_opt, Inf) < 1e-4
            
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            # Use objective-only interface
            if solver_name == "PRIMA-NEWUOA"
                result = PRIMA.newuoa(prob_data.obj_func, prob_data.x0; maxfun=max_iter)
            else
                # Check if problem has finite bounds
                has_bounds = any(prob_data.bl .> -1e20) || any(prob_data.bu .< 1e20)
                if has_bounds
                    result = PRIMA.bobyqa(prob_data.obj_func, prob_data.x0; 
                                        xl=prob_data.bl, xu=prob_data.bu, maxfun=max_iter)
                else
                    # Use large bounds if none specified
                    large_bounds = 1e3
                    result = PRIMA.bobyqa(prob_data.obj_func, prob_data.x0; 
                                        xl=fill(-large_bounds, prob_data.n), 
                                        xu=fill(large_bounds, prob_data.n), maxfun=max_iter)
                end
            end
            x_opt = result[1]
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.reason(result[2]) in [PRIMA.SMALL_TR_RADIUS, PRIMA.FTARGET_ACHIEVED]
            
        elseif solver_name in ["NL-TrustRegion", "NL-LevenbergMarquardt", "NL-GaussNewton"]
            # Use NonlinearSolve interface for NLLS
            n_res(u, p) = prob_data.residual_func(u)
            nl_jac(u, p) = prob_data.jacobian_func(u)
            nl_func = NonlinearFunction(n_res, jac=nl_jac)
            
            prob_nl = NonlinearLeastSquaresProblem(nl_func, prob_data.x0)
            
            if solver_name == "NL-TrustRegion"
                sol = solve(prob_nl, TrustRegion(); maxiters=max_iter)
            elseif solver_name == "NL-LevenbergMarquardt"
                sol = solve(prob_nl, LevenbergMarquardt(); maxiters=max_iter)
            else  # Gauss-Newton
                sol = solve(prob_nl, GaussNewton(); maxiters=max_iter)
            end
            
            x_opt = sol.u
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = SciMLBase.successful_retcode(sol)
            
        elseif solver_name == "TRON"
            # Use JSOSolvers TRON
            # Create a simple NLPModel wrapper
            stats = tron(prob, max_iter=max_iter)
            
            x_opt = stats.solution
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = stats.iter
            converged = stats.status == :first_order

        elseif solver_name == "TRUNK"
            # Use JSOSolvers TRUNK
            # Create a simple NLPModel wrapper
            stats = trunk(prob, max_iter=max_iter)
            
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

function nlsproblems_benchmark(max_problems=20, max_iter=100)
    """Run comprehensive benchmark on NLSProblems.jl problems"""
    
    # Find NLS problems
    problem_names = find_nlls_problems(200)
    
    # Define solvers to test
    solvers = [
        ("QR-NLLS", nonlinearlstr.qr_nlss_bounded_trust_region),
        ("QR-NLLS-scaled", nonlinearlstr.qr_nlss_bounded_trust_region_v2),
        ("Trust-Region", nonlinearlstr.bounded_trust_region),
        ("PRIMA-NEWUOA", nothing),  # Special handling
        ("PRIMA-BOBYQA", nothing),  # Special handling
        ("NL-TrustRegion", nothing),  # Special handling
        ("NL-LevenbergMarquardt", nothing),  # Special handling
        ("NL-GaussNewton", nothing),  # Special handling
        ("TRON", nothing),  # Special handling
        ("TRUNK", nothing)  # Special handling
    ]
    
    results = []
    
    for (i, prob_name) in enumerate(problem_names[1:min(max_problems, length(problem_names))])
        println("\n" * "="^60)
        println("Problem $i/$max_problems: $prob_name")
        
        try
            # Create problem instance
            prob = eval(prob_name)()
            
            # Create Julia functions
            prob_data = create_julia_functions(prob)
            
            println("  Variables: $(prob_data.n)")
            println("  Residuals: $(prob_data.m)")
            
            # Check if problem has bounds
            has_lower_bounds = any(prob_data.bl .> -1e20)
            has_upper_bounds = any(prob_data.bu .< 1e20)
            println("  Lower bounds: $has_lower_bounds")
            println("  Upper bounds: $has_upper_bounds")
            
            initial_obj = prob_data.obj_func(prob_data.x0)
            println("  Initial objective: $initial_obj")
            
            # Skip if initial objective is too large (likely unbounded)
            if initial_obj > 1e10
                println("  Skipping - initial objective too large")
                finalize(prob)
                continue
            end

            if !unconstrained(prob)
                println("  Problem is constrained - skipping")
                finalize(prob)
                continue
            end
            
            # Test each solver
            problem_results = []
            for (solver_name, solver_func) in solvers
                print("    Testing $solver_name... ")
                
                result = test_solver_on_problem(solver_name, solver_func, prob_data, prob, max_iter)
                
                if result.success && result.converged
                    println("✓ obj=$(round(result.final_objective, digits=8)), iters=$(result.iterations)")
                else
                    status = result.success ? "no convergence" : "failed"
                    println("✗ $status")
                end
                
                # Add problem info to result
                result_with_problem = merge(result, (
                    problem = String(prob_name),
                    nvars = prob_data.n,
                    nresiduals = prob_data.m,
                    initial_objective = initial_obj,
                    improvement = initial_obj - result.final_objective,
                    has_bounds = has_lower_bounds || has_upper_bounds
                ))
                
                push!(problem_results, result_with_problem)
            end
            
            append!(results, problem_results)
            
            # Clean up
            finalize(prob)
            
        catch e
            println("  Error with $prob_name: $e")
            continue
        end
    end
    
    return results
end

function analyze_nlsproblems_results(results)
    """Analyze and visualize NLSProblems benchmark results"""
    
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
    println("NLSPROBLEMS.JL BENCHMARK RESULTS")
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
    
    CSV.write("nlsproblems_results.csv", df_with_success)
    println("\nResults saved to nlsproblems_results.csv")
    
    return df_with_success
end

function plot_nlsproblems_performance(df)
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
    axislegend(ax1, position=:rb)
    
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
    
    save("nlsproblems_performance_analysis.png", fig)
    display(fig)
    return fig
end

# Run the benchmark
println("Starting NLSProblems.jl benchmark...")
results = nlsproblems_benchmark(400, 200)  # Test 15 problems, max 200 iterations

# Analyze results
if !isempty(results)
    df_results = analyze_nlsproblems_results(results)
    
    # Create performance plots
    fig = plot_nlsproblems_performance(df_results)
    
    println("\nBenchmark complete!")
    println("Results saved to nlsproblems_results.csv")
    println("Performance plots saved to nlsproblems_performance_profile.png")
else
    println("No results to analyze")
end
