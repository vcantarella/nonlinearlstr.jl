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
Test functions from H.P. Gavin's Levenberg-Marquardt examples, 2011
"""
f_ex1(t::AbstractArray, a) = @. a[1] * exp.(-t / a[2]) + a[3] * t .* exp.(-t / a[4])
function f_ex2(t::AbstractArray, a)
    mt = maximum(t)
    t_norm = t / mt
    return @. a[1] * t_norm + a[2] * t_norm^2 + a[3] * t_norm^3 + a[4] * t_norm^4
end
f_ex3(t::AbstractArray, a) = @. a[1] * exp.(-t / a[2]) + a[3] * sin.(t / a[4])

function create_nonlinearls_test(test_algorithm, ex::Int; kwargs...)
    if ex == 1
        a_true = [ 20.0 , -24.0 , 30 , -40.0 ]
        a0 = [  4.0 , -5.0 , 6.0 ,  10.0 ]
        f = f_ex1
    elseif ex == 2
        a_true = [ 20.0 , 10.0 , 1.0 , 50.0 ]
        a0 = [  5.0 ,  2.0 ,  0.2 ,  10.0 ]
        f = f_ex2
    elseif ex == 3
        a_true = [  6.0 , 20.0 , 1.0 , 5.0 ]
        a0 = [ 10.0 , 50.0 ,  5.0 ,  5.7 ]
        f = f_ex3
    else
        error("Unknown example number")
    end
    # Generate noisy data
    Random.seed!(123)
    t = 0.0:1:100  # Time points
    y_clean = f(t, a_true)
    noise_level = 0.05  # Noise level
    noise = noise_level * randn(length(y_clean)) * mean(y_clean)
    y_data = y_clean + noise
    residual_func(a) = y_data-f(t,a)
    jac_func(a) = ForwardDiff.jacobian(residual_func, a)
    lb = repeat([-Inf], inner=4)
    ub = repeat([Inf], inner=4)
    f_true = residual_func(a_true)
    cost_true = 0.5 * dot(f_true, f_true)  # True cost value
    if test_algorithm != nonlinearlstr.bounded_trust_region
        result = test_algorithm(residual_func, jac_func, a0, lb, ub; kwargs...)
        a_opt, f_opt, g_opt, iter = result
        # Basic convergence tests
        converged = norm(g_opt, Inf) < 1e-4
        bounds_satisfied = all(lb .<= a_opt .<= ub)
        parameter_error = norm(a_opt - a_true)
        cost = 0.5 * dot(f_opt, f_opt)  # Cost function value
    else
        cost_f(a) = 0.5 * dot(residual_func(a), residual_func(a))
        g(a) = ForwardDiff.gradient(cost_f, a)
        H(a) = ForwardDiff.hessian(cost_f, a)
        result = test_algorithm(
            cost_f, g, H, a0, lb, ub; 
            max_iter=100, gtol=1e-8
        )
        a_opt, f_opt, g_opt, iter = result
        # Basic convergence tests
        converged = norm(g_opt, Inf) < 1e-4
        bounds_satisfied = all(lb .<= a_opt .<= ub)
        parameter_error = norm(a_opt - a_true)
        cost = 0.5 * dot(f_opt, f_opt)  # Cost function value
    end

    return (
        a_true = a_true,
        a_opt=a_opt, 
        converged=converged, 
        bounds_satisfied=bounds_satisfied,
        parameter_error=parameter_error,
        iterations=iter,
        initial_cost =0.5 * dot(residual_func(a0), residual_func(a0)),
        cost=cost,
        cost_true=cost_true
    )
end

@testset "Nonlinear Least Squares Solvers Comparison" begin
    
    # Test both solvers on all three examples
    solvers = [
        (nonlinearlstr.qr_nlss_bounded_trust_region, "QR-based TR"),
        (nonlinearlstr.qr_nlss_bounded_trust_region_v2, "QR-based scaled TR"),
        (nonlinearlstr.bounded_trust_region, "Bounded TCG TR"),
        (nonlinearlstr.nlss_bounded_trust_region, "Standard TR")
    ]
    
    example_names = ["Easy", "Medium", "Hard"]
    tolerances = [1e-6, 1e-5, 1e-4]  # Relaxed tolerance for harder problems
    results = []  # Store all results for comparison
    
    for (example_num, (example_name, tol)) in enumerate(zip(example_names, tolerances))
        @testset "Example $example_num: $example_name" begin
            # Create test problem
            
            # Test both solvers
            for (solver_func, solver_name) in solvers
                @testset "$solver_name" begin
                    result = create_nonlinearls_test(solver_func, example_num)
                    result = merge(result, (solver=solver_name, example=example_num, example_name=example_name))

                    # Assertions
                    @test result.converged
                    @test result.bounds_satisfied
                    # @test result.parameter_error < (example_num == 1 ? 0.2 : 0.5)  # Looser for harder problems
                    # @test result.iterations < 100
                    @test result.cost < result.initial_cost * 0.1  # Significant improvement
                    push!(results, result)
                    
                    println("$solver_name - Example $example_num:")
                    println("  True params:  $(round.(result.a_true, digits=1))")
                    println("  Estimated:    $(round.(result.a_opt, digits=2))")
                    println("  Error:        $(round(result.parameter_error, digits=4))")
                    println("  Iterations:   $(result.iterations)")
                    println("  Converged:    $(result.converged)")
                    println("  Cost improvement: $(round(result.cost, digits=3)) vs $(round(result.initial_cost, digits=2))")
                end
            end
        end
    end

    # Make a solver comparison table:
    println("\n" * "="^80)
    println("SOLVER COMPARISON TABLE")
    println("="^80)
    
    # Header
    println(rpad("Solver", 20) * rpad("Example", 10) * rpad("Error", 12) * 
            rpad("Iterations", 12) * rpad("Cost", 12) * rpad("Converged", 10))
    println("-"^80)
    
    # Results by example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num, results)
        for result in example_results
            println(rpad(result.solver, 20) * 
                   rpad(result.example_name, 10) * 
                   rpad(round(result.parameter_error, digits=4), 12) *
                   rpad(result.iterations, 12) *
                   rpad(round(result.cost, digits=6), 12) *
                   rpad(result.converged ? "✓" : "✗", 10))
        end
        if example_num < 3
            println("-"^40)
        end
    end
    
    println("\n" * "="^80)
    println("PERFORMANCE SUMMARY")
    println("="^80)
    
    # Best performer by accuracy for each example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num && r.converged, results)
        if !isempty(example_results)
            best_accuracy = minimum(r -> r.parameter_error, example_results)
            best_solver = filter(r -> r.parameter_error == best_accuracy, example_results)[1]
            println("Example $(example_num) - Best Accuracy: $(best_solver.solver) (Error: $(round(best_accuracy, digits=4)))")
        end
    end
    
    println()
    
    # Best performer by speed for each example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num && r.converged, results)
        if !isempty(example_results)
            fewest_iters = minimum(r -> r.iterations, example_results)
            fastest_solver = filter(r -> r.iterations == fewest_iters, example_results)[1]
            println("Example $(example_num) - Fastest: $(fastest_solver.solver) ($(fewest_iters) iterations)")
        end
    end
    
    # Overall statistics
    println("\n" * "-"^40)
    println("OVERALL STATISTICS")
    println("-"^40)
    
    for solver_name in unique([r.solver for r in results])
        solver_results = filter(r -> r.solver == solver_name, results)
        converged_results = filter(r -> r.converged, solver_results)
        
        if !isempty(converged_results)
            avg_error = mean([r.parameter_error for r in converged_results])
            avg_iters = mean([r.iterations for r in converged_results])
            success_rate = length(converged_results) / length(solver_results) * 100
            
            println("$(rpad(solver_name, 20)): Avg Error=$(round(avg_error, digits=4)), " *
                   "Avg Iters=$(round(avg_iters, digits=1)), Success=$(round(success_rate, digits=1))%")
        else
            println("$(rpad(solver_name, 20)): No successful convergences")
        end
    end
    
    println("="^80)
end


function create_nonlinearsolve_test(test_algorithm, ex::Int; kwargs...)
    if ex == 1
        a_true = [ 20.0 , -24.0 , 30 , -40.0 ]
        a0 = [  4.0 , -5.0 , 6.0 ,  10.0 ]
        f = f_ex1
    elseif ex == 2
        a_true = [ 20.0 , 10.0 , 1.0 , 50.0 ]
        a0 = [  5.0 ,  2.0 ,  0.2 ,  10.0 ]
        f = f_ex2
    elseif ex == 3
        a_true = [  6.0 , 20.0 , 1.0 , 5.0 ]
        a0 = [ 10.0 , 50.0 ,  5.0 ,  5.7 ]
        f = f_ex3
    else
        error("Unknown example number")
    end
    # Generate noisy data
    Random.seed!(123)
    t = 0.0:1:100  # Time points
    y_clean = f(t, a_true)
    noise_level = 0.05  # Noise level
    noise = noise_level * randn(length(y_clean)) * mean(y_clean)
    y_data = y_clean + noise
    residual_func(a, p) = f(t, a)-y_data
    jac_func(a, p) = ForwardDiff.jacobian(x -> residual_func(x, p), a)
    lb = repeat([-Inf], inner=4)
    ub = repeat([Inf], inner=4)
    p = 0
    f_true = residual_func(a_true, p)
    cost_true = 0.5 * dot(f_true, f_true)  # True cost value
    nlprob = NonlinearLeastSquaresProblem(residual_func, a0, p)
    sol = solve(nlprob, test_algorithm; kwargs...)
    a_opt = sol.u
    f_opt = residual_func(a_opt, p)
    g_opt = jac_func(a_opt, p)' * f_opt  # Gradient of 0.
    iter = sol.stats.nsteps

    # Basic convergence tests
    converged = norm(g_opt, Inf) < 1e-4
    bounds_satisfied = all(lb .<= a_opt .<= ub)
    parameter_error = norm(a_opt - a_true)
    cost = 0.5 * dot(f_opt, f_opt)  # Cost function value

    return (
        a_true = a_true,
        a_opt=a_opt, 
        converged=converged, 
        bounds_satisfied=bounds_satisfied,
        parameter_error=parameter_error,
        iterations=iter,
        initial_cost =0.5 * dot(residual_func(a0, p), residual_func(a0, p)),
        cost=cost,
        cost_true=cost_true
    )
end


@testset "QR-Based TR vs Nonlinearsolve solvers" begin
    
    # Test both solvers on all three examples
    solvers = [
        (nonlinearlstr.qr_nlss_bounded_trust_region, "QR-based TR"),
        (nonlinearlstr.qr_nlss_bounded_trust_region_v2, "QR-based scaled TR"),
        (TrustRegion(), "NonlinearSolve TR"),
        (LevenbergMarquardt(), "NonlinearSolve LM"),
        (FastShortcutNLLSPolyalg(), "NonlinearSolve Polyalg"),
    ]
    
    example_names = ["Easy", "Medium", "Hard"]
    tolerances = [1e-6, 1e-5, 1e-4]  # Relaxed tolerance for harder problems
    results = []  # Store all results for comparison
    
    for (example_num, (example_name, tol)) in enumerate(zip(example_names, tolerances))
        @testset "Example $example_num: $example_name" begin
            # Create test problem
            
            # Test both solvers
            for (solver_func, solver_name) in solvers
                @testset "$solver_name" begin
                    if occursin("NonlinearSolve", solver_name)
                        result = create_nonlinearsolve_test(solver_func, example_num;maxiters = 200)
                    else
                        # For non-NLsolve solvers, use the original function    
                        result = create_nonlinearls_test(solver_func, example_num;max_iter = 200)
                    end
                    result = merge(result, (solver=solver_name, example=example_num, example_name=example_name))

                    # Assertions
                    @test result.converged
                    # @test result.bounds_satisfied
                    @test norm(result.a_opt - result.a_true) < 1
                    # @test result.parameter_error < (example_num == 1 ? 0.2 : 0.5)  # Looser for harder problems
                    # @test result.iterations < 100
                    @test result.cost < result.initial_cost * 0.1  # Significant improvement
                    push!(results, result)
                    
                    println("$solver_name - Example $example_num:")
                    println("  True params:  $(round.(result.a_true, digits=1))")
                    println("  Estimated:    $(round.(result.a_opt, digits=2))")
                    println("  Error:        $(round(result.parameter_error, digits=4))")
                    println("  Iterations:   $(result.iterations)")
                    println("  Converged:    $(result.converged)")
                    println("  Cost improvement: $(round(result.cost, digits=3)) vs $(round(result.initial_cost, digits=2))")
                end
            end
        end
    end

    # Make a solver comparison table:
    println("\n" * "="^80)
    println("SOLVER COMPARISON TABLE")
    println("="^80)
    
    # Header
    println(rpad("Solver", 20) * rpad("Example", 10) * rpad("Error", 12) * 
            rpad("Iterations", 12) * rpad("Cost", 12) * rpad("Converged", 10))
    println("-"^80)
    
    # Results by example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num, results)
        for result in example_results
            println(rpad(result.solver, 20) * 
                   rpad(result.example_name, 10) * 
                   rpad(round(result.parameter_error, digits=4), 12) *
                   rpad(result.iterations, 12) *
                   rpad(round(result.cost, digits=6), 12) *
                   rpad(result.converged ? "✓" : "✗", 10))
        end
        if example_num < 3
            println("-"^40)
        end
    end
    
    println("\n" * "="^80)
    println("PERFORMANCE SUMMARY")
    println("="^80)
    
    # Best performer by accuracy for each example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num && r.converged, results)
        if !isempty(example_results)
            best_accuracy = minimum(r -> r.parameter_error, example_results)
            best_solver = filter(r -> r.parameter_error == best_accuracy, example_results)[1]
            println("Example $(example_num) - Best Accuracy: $(best_solver.solver) (Error: $(round(best_accuracy, digits=4)))")
        end
    end
    
    println()
    
    # Best performer by speed for each example
    for example_num in 1:3
        example_results = filter(r -> r.example == example_num && r.converged, results)
        if !isempty(example_results)
            fewest_iters = minimum(r -> r.iterations, example_results)
            fastest_solver = filter(r -> r.iterations == fewest_iters, example_results)[1]
            println("Example $(example_num) - Fastest: $(fastest_solver.solver) ($(fewest_iters) iterations)")
        end
    end
    
    # Overall statistics
    println("\n" * "-"^40)
    println("OVERALL STATISTICS")
    println("-"^40)
    
    for solver_name in unique([r.solver for r in results])
        solver_results = filter(r -> r.solver == solver_name, results)
        converged_results = filter(r -> r.converged, solver_results)
        
        if !isempty(converged_results)
            avg_error = mean([r.parameter_error for r in converged_results])
            avg_iters = mean([r.iterations for r in converged_results])
            success_rate = length(converged_results) / length(solver_results) * 100
            
            println("$(rpad(solver_name, 20)): Avg Error=$(round(avg_error, digits=4)), " *
                   "Avg Iters=$(round(avg_iters, digits=1)), Success=$(round(success_rate, digits=1))%")
        else
            println("$(rpad(solver_name, 20)): No successful convergences")
        end
    end
    
    println("="^80)
end

