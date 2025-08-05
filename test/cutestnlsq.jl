using PRIMA, NonlinearSolve, LinearAlgebra, Statistics, Random, ForwardDiff
using Pkg, Revise, PythonCall, CondaPkg
using DataFrames, CSV, CairoMakie
using Tidier

CondaPkg.add_pip("pycutest")
Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

# Import pycutest
cutest = PythonCall.pyimport("pycutest")

function find_nlls_problems(max_vars=50)
    """Find nonlinear least squares problems with obj='none' (residual form)"""
    # Find problems with objective="none" (these are NLLS problems)
    nlls_problems = cutest.find_problems(
        objective="none",           # NLLS problems have no explicit objective
        n = [1, 50]
    )
    
    return nlls_problems
end

function create_julia_functions(prob)
    """Create Julia function wrappers for pycutest problem"""
    
    # Get problem info
    n = pyconvert(Int, prob.n)
    m = pyconvert(Int, prob.m)  # Number of residuals
    x0 = pyconvert(Vector{Float64}, prob.x0)
    bl = pyconvert(Vector{Float64}, prob.bl)
    bu = pyconvert(Vector{Float64}, prob.bu)

    # For NLLS problems with obj="none", use cons() to get residuals
    residual_func(x) = begin
        r = prob.cons(x, gradient=false)
        return pyconvert(Vector{Float64}, r)
    end
    
    jacobian_func(x) = begin
        r, J = prob.cons(x, gradient=true)
        return pyconvert(Matrix{Float64}, J)
    end
    
    # Create objective as 0.5 * ||r||²
    obj_func(x) = begin
        r = residual_func(x)
        return 0.5 * dot(r, r)
    end
    
    grad_func(x) = begin
        r = residual_func(x)
        J = jacobian_func(x)
        return J' * r
    end
    
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J  # Gauss-Newton approximation
    end
    
    return (
        n=n, m=m, x0=x0, bl=bl, bu=bu,
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        obj_func=obj_func,
        grad_func=grad_func,
        hess_func=hess_func
    )
end

function test_solver_on_problem(solver_name, solver_func, prob_data, max_iter=100)
    """Test a single solver on a problem"""
    try
        start_time = time()
        
        if solver_name in ["QR-NLLS", "Standard-NLLS"]
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
                result = PRIMA.newuoa(prob_data.obj_func, prob_data.x0; maxfun=max_iter*5)
            else
                result = PRIMA.bobyqa(prob_data.obj_func, prob_data.x0; 
                                    xl=prob_data.bl, xu=prob_data.bu, maxfun=max_iter*5)
            end
            x_opt = result[1]
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = result[2].nf
            converged = PRIMA.reason(result[2]) in [PRIMA.SMALL_TR_RADIUS, PRIMA.FTARGET_ACHIEVED]
            
        elseif solver_name in ["NL-TrustRegion", "NL-LevenbergMarquardt"]
            # Use NonlinearSolve interface
            # Define the explicit Jacobian
            prob_fun(u, p) = prob_data.residual_func(u)
            prob_jac(u, p) = prob_data.jacobian_func(u)
            ff = NonlinearFunction(prob_fun, jac=prob_jac)
            prob_nl =  NonlinearLeastSquaresProblem(ff, prob_data.x0)

            if solver_name == "NL-TrustRegion"
                sol = solve(prob_nl, TrustRegion(); maxiters=max_iter, abstol=1e-6)
            else
                sol = solve(prob_nl, LevenbergMarquardt(); maxiters=max_iter, abstol=1e-6)
            end
            
            x_opt = sol.u
            final_obj = prob_data.obj_func(x_opt)
            g_opt = prob_data.grad_func(x_opt)
            iterations = sol.stats.nsteps
            converged = sol.retcode == ReturnCode.Success
            
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
        if e isa PythonError
            throw(e)
        else
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
end

function cutest_nlls_benchmark(max_problems=10, max_iter=100)
    """Run comprehensive benchmark on CUTEst NLLS problems"""
    
    # Find NLLS problems
    problems = find_nlls_problems(max_problems)
    
    # Define solvers to test
    solvers = [
        ("QR-NLLS", nonlinearlstr.qr_nlss_bounded_trust_region),
        ("Standard-NLLS", nonlinearlstr.nlss_bounded_trust_region),
        ("Trust-Region", nonlinearlstr.bounded_trust_region),
        ("PRIMA-NEWUOA", nothing),  # Special handling
        ("PRIMA-BOBYQA", nothing),  # Special handling
        ("NL-TrustRegion", nothing),  # Special handling
        ("NL-LevenbergMarquardt", nothing)  # Special handling
    ]
    
    results = []

    for (i, prob_name) in enumerate([problems[i] for i in collect(1:min(max_problems, length(problems)))])
        println("\n" * "="^60)
        println("Problem $i/$max_problems: $prob_name")
        # Import problem
        if pyconvert(Bool, prob_name != pystr("MSQRTA"))
            prob = cutest.import_problem(prob_name) #sifParams=pydict(Dict("N"=>50)))
        else
            continue
        end

        # Create Julia functions
        prob_data = create_julia_functions(prob)

        if prob_data.n > 200
            continue  # Skip problems with too many variables
        end
        println("  Variables: $(prob_data.n)")
        println("  Residuals: $(prob_data.m)")
        println("  Bounds: $(any(prob_data.bl .> -1e20) || any(prob_data.bu .< 1e20))")
        
        initial_obj = prob_data.obj_func(prob_data.x0)
        println("  Initial objective: $initial_obj")
            # Test each solver
            problem_results = []
            for (solver_name, solver_func) in solvers
                print("    Testing $solver_name... ")
                
                result = test_solver_on_problem(solver_name, solver_func, prob_data, max_iter)
                
                # if result.success && result.converged
                #     println("✓ obj=$(round(result.final_objective, digits=6)), iters=$(result.iterations)")
                # else
                #     println("✗ failed")
                # end
                
                # Add problem info to result
                result_with_problem = merge(result, (
                    problem = prob_name,
                    nvars = prob_data.n,
                    nresiduals = prob_data.m,
                    initial_objective = initial_obj,
                    improvement = initial_obj - result.final_objective
                ))
                
                push!(problem_results, result_with_problem)
            end
            
            append!(results, problem_results)
            
            # Clean up
            #prob.close()
    end
    
    return results
end

# Run the benchmark
println("Starting CUTEst NLLS benchmark...")
results = cutest_nlls_benchmark(100, 100)  # Test 100 problems, max 100 iterations

# Analyze results
df_results = DataFrame(results)

using Tidier
# summarize the results per solver
df_summary = @chain(df_results,
    @group_by(solver),
    @summarize(
        success = sum(success),
        converged = sum(converged),
        avg_objective = median(final_objective),
        avg_iterations = median(iterations),
        avg_time = median(time),
        avg_gradient_norm = median(final_gradient_norm),
        bounds_satisfied = sum(bounds_satisfied),
        avg_improvement = median(improvement)
    )
)

# Create performance plot
function plot_performance_profile(df)
    successful = filter(r -> r.success && r.converged, df)
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Performance Ratio (log scale)", ylabel="Fraction of Problems Solved",
              title="Performance Profile - CUTEst NLLS Benchmark")
    
    solvers = unique(successful.solver)
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink]
    
    for (i, solver) in enumerate(solvers)
        solver_data = filter(r -> r.solver == solver, successful)
        
        # Performance profile based on iterations
        ratios = Float64[]
        for prob in unique(solver_data.problem)
            prob_results = filter(r -> string(r.problem) == string(prob), successful)
            if nrow(prob_results) > 1
                solver_result = filter(r -> r.solver == solver && string(r.problem) == string(prob), prob_results)
                if !isempty(solver_result)
                    best_iters = minimum([r.iterations for r in eachrow(prob_results)])
                    ratio = solver_result[1, :iterations] / best_iters
                    push!(ratios, ratio)
                end
            end
        end
        
        if !isempty(ratios)
            sorted_ratios = sort(ratios)
            n = length(sorted_ratios)
            fractions = (1:n) / n
            
            lines!(ax, log10.(sorted_ratios), fractions, 
                  color=colors[i], linewidth=2, label=solver)
        end
    end
    
    axislegend(ax, position=:rb)
    save("cutest_performance_profile.png", fig)
    return fig
end

# Create performance plot
if nrow(df_results) > 0
    fig = plot_performance_profile(df_results)
    display(fig)
end