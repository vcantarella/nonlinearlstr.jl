using Test
using LinearAlgebra
using Statistics
using Random
using ForwardDiff
using Pkg
using Revise
using NonlinearSolve

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

"""
Test functions from H.P. Gavin's Levenberg-Marquardt examples
"""

function lm_func(t::AbstractVector, a::AbstractVector, example_number::Int)
    """Three example functions for nonlinear least squares curve-fitting"""
    if example_number == 1
        # Example 1: easy for LM ... a poor initial guess is ok
        y_hat = a[1] * exp.(-t / a[2]) + a[3] * t .* exp.(-t / a[4])
    elseif example_number == 2
        # Example 2: medium for LM ... local minima
        mt = maximum(t)
        t_norm = t / mt
        y_hat = a[1] * t_norm + a[2] * t_norm.^2 + a[3] * t_norm.^3 + a[4] * t_norm.^4
    elseif example_number == 3
        # Example 3: difficult for LM ... needs a very good initial guess
        y_hat = a[1] * exp.(-t / a[2]) + a[3] * sin.(t / a[4])
    else
        error("example_number must be 1, 2, or 3")
    end
    return y_hat
end

function create_test_problem(example_number::Int, noise_level::Float64 = 0.05)
    """Create complete test problem with data and functions"""
    # Time points
    t = collect(0.0:1:100.0)
    
    # True parameters for each example
    a_true = if example_number == 1
        [ 20.0 , -24.0 , 30 , -40.0 ]  # Easy case
    elseif example_number == 2
        [ 20.0 , 10.0 , 1.0 , 50.0 ]  # Medium case
    else
        [  6.0 , 20.0 , 1.0 , 5.0 ]  # Hard case
    end
    
    # Generate noisy data
    Random.seed!(123)
    y_clean = lm_func(t, a_true, example_number)
    noise = noise_level * randn(length(y_clean)) * mean(y_clean)
    y_data = y_clean + noise
    
    # Create residual and Jacobian functions
    residual_func(a) = y_data - lm_func(t, a, example_number)
    jac_func(a) = ForwardDiff.jacobian(residual_func, a)
    
    return (t=t, y_data=y_data, a_true=a_true, residual_func=residual_func, jac_func=jac_func)
end
function get_initial_conditions(example_number::Int)
    """Get appropriate initial guess and bounds for each example"""
    if example_number == 1
        # True: [20.0, -24.0, 30.0, -40.0]
        return ([  4.0 , -5.0 , 6.0 ,  10.0 ], repeat([-Inf], inner=4), repeat([Inf], inner=4))
    elseif example_number == 2
        # True: [20.0, 10.0, 1.0, 50.0]
        return ([  5.0 ,  2.0 ,  0.2 ,  10.0 ], repeat([-Inf], inner=4), repeat([Inf], inner=4))
    else
        # True: [6.0, 20.0, 1.0, 5.0]
        return ([ 10.0 , 50.0 ,  5.0 ,  5.7 ], repeat([-Inf], inner=4), repeat([Inf], inner=4))
    end
end

function test_solver(solver_func, solver_name, problem, a0, lb, ub; kwargs...)
    """Test a single solver on a problem"""
    result = solver_func(problem.residual_func, problem.jac_func, a0, lb, ub; kwargs...)
    a_opt, f_opt, g_opt, iter = result
    
    # Basic convergence tests
    converged = norm(g_opt, Inf) < 1e-4
    bounds_satisfied = all(lb .<= a_opt .<= ub)
    parameter_error = norm(a_opt - problem.a_true)
    cost = 0.5 * dot(f_opt, f_opt)  # Cost function value
    
    return (
        a_opt=a_opt, 
        converged=converged, 
        bounds_satisfied=bounds_satisfied,
        parameter_error=parameter_error,
        iterations=iter,
        solver=solver_name,
        cost=cost
    )
end

@testset "Nonlinear Least Squares Solvers Comparison" begin
    
    # Test both solvers on all three examples
    solvers = [
        (nonlinearlstr.qr_nlss_bounded_trust_region, "QR-based TR"),
        #(nonlinearlstr.nlss_bounded_trust_region, "Standard TR")
    ]
    
    example_names = ["Easy", "Medium", "Hard"]
    tolerances = [1e-6, 1e-5, 1e-4]  # Relaxed tolerance for harder problems
    
    for (example_num, (example_name, tol)) in enumerate(zip(example_names, tolerances))
        @testset "Example $example_num: $example_name" begin
            # Create test problem
            problem = create_test_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            initial_residuals = problem.residual_func(a0)
            initial_cost = 0.5 * dot(initial_residuals, initial_residuals)

            results = []
            
            # Test both solvers
            for (solver_func, solver_name) in solvers
                @testset "$solver_name" begin
                    result = test_solver(
                        solver_func, solver_name, problem, a0, lb, ub;
                        max_iter=100, gtol=1e-8,
                    )
                    
                    # Assertions
                    @test result.converged
                    @test result.bounds_satisfied
                    @test result.parameter_error < (example_num == 1 ? 0.2 : 0.5)  # Looser for harder problems
                    @test result.iterations < 100
                    @test result.cost < initial_cost * 0.1  # Significant improvement
                    push!(results, result)
                    
                    println("$solver_name - Example $example_num:")
                    println("  True params:  $(round.(problem.a_true, digits=3))")
                    println("  Estimated:    $(round.(result.a_opt, digits=3))")
                    println("  Error:        $(round(result.parameter_error, digits=4))")
                    println("  Iterations:   $(result.iterations)")
                    println("  Converged:    $(result.converged)")
                    println("  Cost improvement: $(round(result.cost, digits=3)) vs $(round(initial_cost, digits=2))")
                end
            end
            
            # Compare solvers
            if length(results) == 2
                qr_result, std_result = results
                println("Comparison for Example $example_num ($example_name):")
                println("  QR accuracy:     $(round(qr_result.parameter_error, digits=4))")
                println("  Standard accuracy: $(round(std_result.parameter_error, digits=4))")
                println("  QR iterations:   $(qr_result.iterations)")
                println("  Standard iterations: $(std_result.iterations)")
                println("  Winner (accuracy): $(qr_result.parameter_error < std_result.parameter_error ? "QR" : "Standard")")
                println("  Winner (speed):    $(qr_result.iterations < std_result.iterations ? "QR" : "Standard")")
                println("-" ^ 50)
            end
        end
    end
end

# @testset "Solver Robustness Tests" begin
#     """Additional robustness tests"""
    
#     @testset "High noise tolerance" begin
#         # Test with higher noise levels
#         problem = create_test_problem(1, 0.1)  # 10% noise
#         a0, lb, ub = get_initial_conditions(1)
        
#         for (solver_func, solver_name) in solvers
#             result = test_solver(
#                 solver_func, solver_name, problem, a0, lb, ub;
#                 max_iter=500, gtol=1e-6
#             )
#             @test result.bounds_satisfied
#             # Relaxed convergence for high noise
#         end
#     end
    
#     @testset "Poor initial guess" begin
#         # Test Example 1 with very poor initial guess
#         problem = create_test_problem(1, 0.01)
#         a0 = [0.1, 0.1, 0.1, 0.1]  # Very poor guess
#         lb, ub = [0.01, 0.01, 0.01, 0.01], [20.0, 20.0, 20.0, 20.0]
        
#         for (solver_func, solver_name) in solvers
#             result = test_solver(
#                 solver_func, solver_name, problem, a0, lb, ub;
#                 max_iter=500, gtol=1e-6
#             )
#             @test result.bounds_satisfied
#             println("$solver_name with poor guess: error = $(round(result.parameter_error, digits=4))")
#         end
#     end
# end

function create_cost_problem(example_number::Int, noise_level::Float64 = 0.05)
    """Create test problem with explicit cost function and Hessian for bounded_trust_region"""
    # Time points
    t = collect(0.0:1:100.0)
    
    # True parameters for each example
    a_true = if example_number == 1
        [ 20.0 , -24.0 , 30 , -40.0 ]  # Easy case
    elseif example_number == 2
        [ 20.0 , 10.0 , 1.0 , 50.0 ]  # Medium case
    else
        [  6.0 , 20.0 , 1.0 , 5.0 ]  # Hard case
    end
    # Generate noisy data
    Random.seed!(123)
    y_clean = lm_func(t, a_true, example_number)
    noise = noise_level * randn(length(y_clean)) * mean(y_clean)
    y_data = y_clean + noise
    
    # Cost function: f(a) = 0.5 * ||residuals||²
    function cost_func(a)
        residuals = y_data - lm_func(t, a, example_number)
        return 0.5 * dot(residuals, residuals)
    end
    
    # Gradient function: ∇f(a) = J^T * residuals
    function grad_func(a)
        residuals = y_data - lm_func(t, a, example_number)
        jac = ForwardDiff.jacobian(x -> lm_func(t, x, example_number), a)
        return -jac' * residuals  # Note: negative because residuals = y_data - model
    end
    
    # Hessian function: H(a) = J^T * J + higher-order terms
    # For Gauss-Newton approximation: H ≈ J^T * J
    function hessian_func(a)
        jac = ForwardDiff.jacobian(x -> lm_func(t, x, example_number), a)
        return jac' * jac
    end
    
    # Full Hessian (including second-order terms) - more accurate but expensive
    function full_hessian_func(a)
        return ForwardDiff.hessian(cost_func, a)
    end
    
    return (
        t=t, 
        y_data=y_data, 
        a_true=a_true, 
        cost_func=cost_func,
        grad_func=grad_func,
        hessian_func=hessian_func,  # Gauss-Newton approximation
        full_hessian_func=full_hessian_func  # Full Hessian
    )
end

function test_cost_solver(solver_func, solver_name, problem, a0, lb, ub; use_full_hessian=false, kwargs...)
    """Test bounded_trust_region solver with cost function interface"""
    
    hess_func = use_full_hessian ? problem.full_hessian_func : problem.hessian_func
    hess_name = use_full_hessian ? "Full Hessian" : "Gauss-Newton"
    
    result = solver_func(
        problem.cost_func, 
        problem.grad_func, 
        hess_func,
        a0, lb, ub; 
        kwargs...
    )
    
    # Extract results (format may vary depending on your bounded_trust_region output)
    if length(result) == 4
        a_opt, f_opt, g_opt, iter = result
    else
        # Adjust based on your actual output format
        a_opt, iter = result[1], result[end]
        f_opt = problem.cost_func(a_opt)
        g_opt = problem.grad_func(a_opt)
    end
    
    # Basic convergence tests
    converged = norm(g_opt, Inf) < 1e-4
    bounds_satisfied = all(lb .<= a_opt .<= ub)
    parameter_error = norm(a_opt - problem.a_true)
    
    return (
        a_opt=a_opt, 
        converged=converged, 
        bounds_satisfied=bounds_satisfied,
        parameter_error=parameter_error,
        iterations=iter,
        solver="$solver_name ($hess_name)",
        cost_value=f_opt
    )
end

@testset "Trust Region with Explicit Cost/Hessian" begin
    
    # Test bounded_trust_region with both Hessian types
    cost_solvers = [
        (nonlinearlstr.bounded_trust_region, "Bounded TR")
    ]
    
    hessian_types = [false, true]  # Gauss-Newton vs Full Hessian
    hessian_names = ["Gauss-Newton", "Full Hessian"]
    
    example_names = ["Easy", "Medium", "Hard"]
    
    for (example_num, example_name) in enumerate(example_names)
        @testset "Example $example_num: $example_name (Cost Interface)" begin
            # Create test problem with cost function interface
            problem = create_cost_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            
            all_results = []
            
            # Test with both Hessian approximations
            for (use_full_hessian, hess_name) in zip(hessian_types, hessian_names)
                for (solver_func, solver_name) in cost_solvers
                    @testset "$solver_name with $hess_name" begin
                        result = test_cost_solver(
                            solver_func, solver_name, problem, a0, lb, ub;
                            use_full_hessian=use_full_hessian,
                            max_iter=300, gtol=1e-8
                        )
                        
                        # Assertions
                        @test result.bounds_satisfied
                        # Note: May need to relax convergence for some cases
                        if result.converged
                            @test result.parameter_error < (example_num == 1 ? 0.5 : 1.0)
                        end
                        @test result.iterations < 250
                        
                        push!(all_results, result)
                        
                        println("$(result.solver) - Example $example_num:")
                        println("  True params:  $(round.(problem.a_true, digits=3))")
                        println("  Estimated:    $(round.(result.a_opt, digits=3))")
                        println("  Error:        $(round(result.parameter_error, digits=4))")
                        println("  Cost value:   $(round(result.cost_value, digits=6))")
                        println("  Iterations:   $(result.iterations)")
                        println("  Converged:    $(result.converged)")
                        println()
                    end
                end
            end
            
            # Compare Gauss-Newton vs Full Hessian
            if length(all_results) >= 2
                gn_result = all_results[1]  # Gauss-Newton
                fh_result = all_results[2]  # Full Hessian
                
                println("Hessian Comparison for Example $example_num:")
                println("  Gauss-Newton error:  $(round(gn_result.parameter_error, digits=4))")
                println("  Full Hessian error:  $(round(fh_result.parameter_error, digits=4))")
                println("  Gauss-Newton iters:  $(gn_result.iterations)")
                println("  Full Hessian iters:  $(fh_result.iterations)")
                println("  Better accuracy:     $(gn_result.parameter_error < fh_result.parameter_error ? "Gauss-Newton" : "Full Hessian")")
                println("  Faster convergence:  $(gn_result.iterations < fh_result.iterations ? "Gauss-Newton" : "Full Hessian")")
                println("-" ^ 50)
            end
        end
    end
end

@testset "Complete Solver Comparison" begin
    """Compare all three solver types on the same problems"""
    
    for example_num in 1:3
        @testset "Complete comparison - Example $example_num" begin
            # Create both problem types
            nlls_problem = create_test_problem(example_num, 0.02)
            cost_problem = create_cost_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            
            results = []
            
            # Test NLLS solvers
            nlls_solvers = [
                (nonlinearlstr.qr_nlss_bounded_trust_region, "QR-NLLS"),
                (nonlinearlstr.nlss_bounded_trust_region, "Standard-NLLS")
            ]
            
            for (solver_func, solver_name) in nlls_solvers
                result = test_solver(
                    solver_func, solver_name, nlls_problem, a0, lb, ub;
                    max_iter=300, gtol=1e-8
                )
                push!(results, result)
            end
            
            # Test cost-based solver with Gauss-Newton Hessian
            cost_result = test_cost_solver(
                nonlinearlstr.bounded_trust_region, "Cost-based", 
                cost_problem, a0, lb, ub;
                use_full_hessian=false, max_iter=300, gtol=1e-8
            )
            push!(results, cost_result)
            
            # Summary comparison
            println("\nComplete Comparison - Example $example_num:")
            println("="^60)
            for result in results
                println("$(rpad(result.solver, 15)): Error=$(round(result.parameter_error, digits=4)), Iters=$(result.iterations), Conv=$(result.converged)")
            end
            
            # Find best performer
            converged_results = filter(r -> r.converged, results)
            if !isempty(converged_results)
                best_accuracy = minimum(r -> r.parameter_error, converged_results)
                best_solver = findfirst(r -> r.parameter_error == best_accuracy, converged_results)
                println("Best accuracy: $(converged_results[best_solver].solver)")
                
                fastest = minimum(r -> r.iterations, converged_results)
                fastest_solver = findfirst(r -> r.iterations == fastest, converged_results)
                println("Fastest convergence: $(converged_results[fastest_solver].solver)")
            end
            println("-" ^ 60)
        end
    end
end

function comprehensive_benchmark(max_runs::Int = 3)
    """Benchmark all solver types"""
    println("\n" * "="^80)
    println("COMPREHENSIVE PERFORMANCE BENCHMARK")
    println("="^80)
    
    for example_num in 1:3
        println("\nExample $example_num:")
        println("-" ^ 40)
        
        # Storage for timing results
        times = Dict(
            "QR-NLLS" => Float64[],
            "Standard-NLLS" => Float64[],
            "Cost-GN" => Float64[],
            "Cost-Full" => Float64[]
        )
        
        for run in 1:max_runs
            nlls_problem = create_test_problem(example_num, 0.02)
            cost_problem = create_cost_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            
            # Benchmark QR-NLLS
            time_start = time()
            nonlinearlstr.qr_nlss_bounded_trust_region(
                nlls_problem.residual_func, nlls_problem.jac_func, a0, lb, ub;
                max_iter=300, gtol=1e-8
            )
            push!(times["QR-NLLS"], time() - time_start)
            
            # Benchmark Standard-NLLS
            time_start = time()
            nonlinearlstr.nlss_bounded_trust_region(
                nlls_problem.residual_func, nlls_problem.jac_func, a0, lb, ub;
                max_iter=300, gtol=1e-8
            )
            push!(times["Standard-NLLS"], time() - time_start)
            
            # Benchmark Cost-based with Gauss-Newton
            time_start = time()
            nonlinearlstr.bounded_trust_region(
                cost_problem.cost_func, cost_problem.grad_func, 
                cost_problem.hessian_func, a0, lb, ub;
                max_iter=300, gtol=1e-8
            )
            push!(times["Cost-GN"], time() - time_start)
            
            # Benchmark Cost-based with Full Hessian
            time_start = time()
            nonlinearlstr.bounded_trust_region(
                cost_problem.cost_func, cost_problem.grad_func, 
                cost_problem.full_hessian_func, a0, lb, ub;
                max_iter=300, gtol=1e-8
            )
            push!(times["Cost-Full"], time() - time_start)
        end
        
        # Report results
        for (solver_name, solver_times) in times
            avg_time = mean(solver_times) * 1000  # Convert to ms
            println("  $(rpad(solver_name, 15)): $(round(avg_time, digits=1)) ms (avg)")
        end
        
        # Calculate relative performance
        baseline = mean(times["Standard-NLLS"])
        for (solver_name, solver_times) in times
            if solver_name != "Standard-NLLS"
                speedup = baseline / mean(solver_times)
                println("  $(rpad(solver_name, 15)) speedup: $(round(speedup, digits=2))x")
            end
        end
    end
    println("="^80)
end

# Uncomment to run benchmark
# benchmark_comparison()

# Uncomment to run comprehensive benchmark
# comprehensive_benchmark()

function create_nonlinearsolve_problem(example_number::Int, noise_level::Float64 = 0.05)
    """Create test problem compatible with NonlinearSolve.jl interface"""
    # Time points
    t = collect(0.0:0.1:5.0)
    
    # True parameters for each example
    a_true = if example_number == 1
        [ 20.0 , -24.0 , 30 , -40.0 ]  # Easy case
    elseif example_number == 2
        [ 20.0 , 10.0 , 1.0 , 50.0 ]  # Medium case
    else
        [  6.0 , 20.0 , 1.0 , 5.0 ]  # Hard case
    end
    
    # Generate noisy data
    Random.seed!(123)
    y_clean = lm_func(t, a_true, example_number)
    noise = noise_level * randn(length(y_clean)) * mean(y_clean)
    y_data = y_clean + noise
    
    # NonlinearSolve expects f(u, p) = 0 format
    # For least squares: f(u, p) = residuals = y_data - model(u)
    function nonlinearsolve_func(u, p)
        model_pred = lm_func(t, u, example_number)
        residuals = y_data - model_pred
        return residuals
    end
    
    return (
        t=t, 
        y_data=y_data, 
        a_true=a_true, 
        nl_func=nonlinearsolve_func,
        # Also keep the original functions for comparison
        residual_func = (a) -> y_data - lm_func(t, a, example_number),
        jac_func = (a) -> ForwardDiff.jacobian(x -> y_data - lm_func(t, x, example_number), a)
    )
end

function test_nonlinearsolve_solver(solver_algorithm, solver_name, problem, a0, lb, ub; kwargs...)
    """Test NonlinearSolve.jl solver on least squares problem"""
    
    # Create NonlinearProblem
    nl_prob = NonlinearLeastSquaresProblem(problem.nl_func, a0)
    
    # Solve with specified algorithm
    try
        sol = solve(nl_prob, solver_algorithm; kwargs...)
        
        a_opt = sol.u
        iter = sol.stats.nsteps
        converged = sol.retcode == ReturnCode.Success
        
        # Calculate residuals and gradient at solution
        f_opt = problem.residual_func(a_opt)
        g_opt = problem.jac_func(a_opt)' * f_opt  # Gradient of 0.5||f||²
        
        # Basic convergence tests
        bounds_satisfied = all(lb .<= a_opt .<= ub)
        parameter_error = norm(a_opt - problem.a_true)
        cost = 0.5 * dot(f_opt, f_opt)
        
        return (
            a_opt=a_opt, 
            converged=converged, 
            bounds_satisfied=bounds_satisfied,
            parameter_error=parameter_error,
            iterations=iter,
            solver=solver_name,
            cost=cost,
            success=true
        )
        
    catch e
        println("NonlinearSolve solver $solver_name failed: $e")
        return (
            a_opt=a0, 
            converged=false, 
            bounds_satisfied=false,
            parameter_error=Inf,
            iterations=0,
            solver=solver_name,
            cost=Inf,
            success=false
        )
    end
end

@testset "QR vs NonlinearSolve Comparison" begin
    
    # Define solvers to compare
    qr_solvers = [
        (nonlinearlstr.qr_nlss_bounded_trust_region, "QR-based TR")
    ]
    
    # NonlinearSolve algorithms to test
    nl_solvers = [
        (TrustRegion(), "NonlinearSolve TR"),
        (GaussNewton(), "NonlinearSolve Newton"),
        (LevenbergMarquardt(), "NonlinearSolve LM"),
        (FastShortcutNLLSPolyalg(), "NonlinearSolve FastShortcut")
    ]
    
    example_names = ["Easy", "Medium", "Hard"]
    
    for (example_num, example_name) in enumerate(example_names)
        @testset "QR vs NonlinearSolve - Example $example_num: $example_name" begin
            # Create test problems
            qr_problem = create_test_problem(example_num, 0.02)
            nl_problem = create_nonlinearsolve_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            
            initial_residuals = qr_problem.residual_func(a0)
            initial_cost = 0.5 * dot(initial_residuals, initial_residuals)
            
            all_results = []
            
            # Test QR solver
            for (solver_func, solver_name) in qr_solvers
                @testset "$solver_name" begin
                    result = test_solver(
                        solver_func, solver_name, qr_problem, a0, lb, ub;
                        max_iter=200, gtol=1e-8
                    )
                    
                    @test result.converged
                    @test result.bounds_satisfied
                    @test result.cost < initial_cost * 0.1
                    
                    push!(all_results, result)
                    
                    println("$solver_name - Example $example_num:")
                    println("  True params:  $(round.(qr_problem.a_true, digits=3))")
                    println("  Estimated:    $(round.(result.a_opt, digits=3))")
                    println("  Error:        $(round(result.parameter_error, digits=4))")
                    println("  Iterations:   $(result.iterations)")
                    println("  Cost:         $(round(result.cost, digits=6))")
                    println("  Converged:    $(result.converged)")
                    println()
                end
            end
            
            # Test NonlinearSolve solvers
            for (solver_alg, solver_name) in nl_solvers
                @testset "$solver_name" begin
                    result = test_nonlinearsolve_solver(
                        solver_alg, solver_name, nl_problem, a0, lb, ub;
                        maxiters=200, abstol=1e-12, reltol=1e-12
                    )
                    
                    if result.success
                        # Only test if solver succeeded
                        @test result.bounds_satisfied
                        if result.converged
                            @test result.cost < initial_cost * 0.1
                        end
                        
                        push!(all_results, result)
                        
                        println("$solver_name - Example $example_num:")
                        println("  True params:  $(round.(nl_problem.a_true, digits=3))")
                        println("  Estimated:    $(round.(result.a_opt, digits=3))")
                        println("  Error:        $(round(result.parameter_error, digits=4))")
                        println("  Iterations:   $(result.iterations)")
                        println("  Cost:         $(round(result.cost, digits=6))")
                        println("  Converged:    $(result.converged)")
                        println()
                    else
                        println("$solver_name - Example $example_num: FAILED")
                        println()
                    end
                end
            end
            
            # Compare all successful solvers
            successful_results = filter(r -> haskey(r, :success) ? r.success && r.converged : r.converged, all_results)
            
            if length(successful_results) >= 2
                println("Solver Comparison for Example $example_num ($example_name):")
                println("="^70)
                
                # Sort by accuracy
                sorted_by_accuracy = sort(successful_results, by = r -> r.parameter_error)
                println("Ranking by Accuracy:")
                for (i, result) in enumerate(sorted_by_accuracy)
                    println("  $i. $(rpad(result.solver, 20)): Error = $(round(result.parameter_error, digits=4))")
                end
                println()
                
                # Sort by iterations (speed)
                sorted_by_speed = sort(successful_results, by = r -> r.iterations)
                println("Ranking by Speed (iterations):")
                for (i, result) in enumerate(sorted_by_speed)
                    println("  $i. $(rpad(result.solver, 20)): $(result.iterations) iterations")
                end
                println()
                
                # Sort by final cost
                sorted_by_cost = sort(successful_results, by = r -> r.cost)
                println("Ranking by Final Cost:")
                for (i, result) in enumerate(sorted_by_cost)
                    println("  $i. $(rpad(result.solver, 20)): Cost = $(round(result.cost, digits=6))")
                end
                
                println("-" ^ 70)
            end
        end
    end
end

# Add benchmark function for NonlinearSolve comparison
function benchmark_qr_vs_nonlinearsolve(max_runs::Int = 3)
    """Benchmark QR solver against NonlinearSolve algorithms"""
    println("\n" * "="^80)
    println("QR vs NONLINEARSOLVE PERFORMANCE BENCHMARK")
    println("="^80)
    
    # Define solvers
    all_solvers = [
        ("QR", (prob, a0, lb, ub) -> nonlinearlstr.qr_nlss_bounded_trust_region(
            prob.residual_func, prob.jac_func, a0, lb, ub; max_iter=200, gtol=1e-8)),
        ("NL-TR", (prob, a0, lb, ub) -> solve(NonlinearProblem(prob.nl_func, a0), TrustRegion(); maxiters=200)),
        ("NL-LM", (prob, a0, lb, ub) -> solve(NonlinearProblem(prob.nl_func, a0), LevenbergMarquardt(); maxiters=200)),
        ("NL-Newton", (prob, a0, lb, ub) -> solve(NonlinearProblem(prob.nl_func, a0), NewtonRaphson(); maxiters=200))
    ]
    
    for example_num in 1:3
        println("\nExample $example_num:")
        println("-" ^ 40)
        
        # Storage for timing and success results
        times = Dict{String, Vector{Float64}}()
        successes = Dict{String, Int}()
        errors = Dict{String, Vector{Float64}}()
        
        for (solver_name, _) in all_solvers
            times[solver_name] = Float64[]
            successes[solver_name] = 0
            errors[solver_name] = Float64[]
        end
        
        for run in 1:max_runs
            qr_problem = create_test_problem(example_num, 0.02)
            nl_problem = create_nonlinearsolve_problem(example_num, 0.02)
            a0, lb, ub = get_initial_conditions(example_num)
            
            for (solver_name, solver_func) in all_solvers
                try
                    time_start = time()
                    if solver_name == "QR"
                        result = solver_func(qr_problem, a0, lb, ub)
                        a_opt, f_opt, g_opt, iter = result
                        success = norm(g_opt, Inf) < 1e-4
                        error_val = norm(a_opt - qr_problem.a_true)
                    else
                        result = solver_func(nl_problem, a0, lb, ub)
                        success = result.retcode == ReturnCode.Success
                        error_val = success ? norm(result.u - nl_problem.a_true) : Inf
                    end
                    
                    elapsed = time() - time_start
                    push!(times[solver_name], elapsed)
                    
                    if success
                        successes[solver_name] += 1
                        push!(errors[solver_name], error_val)
                    else
                        push!(errors[solver_name], Inf)
                    end
                    
                catch e
                    println("  $solver_name failed on run $run: $e")
                    push!(times[solver_name], Inf)
                    push!(errors[solver_name], Inf)
                end
            end
        end
        
        # Report results
        println("Performance Results:")
        for (solver_name, solver_times) in times
            valid_times = filter(t -> t != Inf, solver_times)
            valid_errors = filter(e -> e != Inf, errors[solver_name])
            
            if !isempty(valid_times)
                avg_time = mean(valid_times) * 1000  # Convert to ms
                avg_error = isempty(valid_errors) ? Inf : mean(valid_errors)
                success_rate = successes[solver_name] / max_runs
                
                println("  $(rpad(solver_name, 10)): $(round(avg_time, digits=1)) ms, " *
                       "Error: $(round(avg_error, digits=4)), " *
                       "Success: $(round(success_rate*100, digits=1))%")
            else
                println("  $(rpad(solver_name, 10)): All runs failed")
            end
        end
        
        # Find best performers among successful runs
        successful_solvers = filter(name -> successes[name] > 0, collect(keys(successes)))
        if !isempty(successful_solvers)
            # Best accuracy
            best_accuracy_solver = argmin(name -> mean(filter(e -> e != Inf, errors[name])), successful_solvers)
            best_accuracy = mean(filter(e -> e != Inf, errors[best_accuracy_solver]))
            
            # Fastest
            fastest_solver = argmin(name -> mean(filter(t -> t != Inf, times[name])), successful_solvers)
            fastest_time = mean(filter(t -> t != Inf, times[fastest_solver])) * 1000
            
            println("  Best accuracy: $best_accuracy_solver ($(round(best_accuracy, digits=4)))")
            println("  Fastest: $fastest_solver ($(round(fastest_time, digits=1)) ms)")
        end
    end
    println("="^80)
end

# Uncomment to run the benchmark
# benchmark_qr_vs_nonlinearsolve()
