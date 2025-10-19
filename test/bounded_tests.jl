using NLSProblems
using JSOSolvers
using PRIMA
using NonlinearSolve
using NLPModels: bound_constrained, finalize, AbstractNLSModel, residual, jac_residual, obj, grad
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using Tidier
using LinearAlgebra, Statistics
using PythonCall
scipy = pyimport("scipy")

using nonlinearlstr

function find_bounded_problems(max_vars=Inf)
    """Find bounded NLS problems from NLSProblems.jl package"""
    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])
    
    valid_problems = []
    
    for prob_name in all_problems
            prob = eval(prob_name)()
            
            # Filter by size and check if it's a valid NLS problem
            if !bound_constrained(prob)
                println("  Problem $prob_name is not bound constrained, skipping")
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
    
    println("Found $(length(valid_problems)) valid bounded NLS problems (≤ $max_vars variables)")
    return valid_problems
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

probs = find_bounded_problems()

nls_functions = [create_nls_functions(eval(prob_name)()) for prob_name in probs]

prob_data = nls_functions[1]  # Use the first problem for testing
result = nonlinearlstr.lm_double_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=300, gtol=1e-8
            )
result = nonlinearlstr.lm_interior_and_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=300, gtol=1e-8
)

result = nonlinearlstr.lm_trust_region_reflective_v2(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)

result = nonlinearlstr.active_set_svd_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)

pyresult = scipy.optimize.least_squares(prob_data.residual_func, prob_data.x0, jac=prob_data.jacobian_func, bounds=(prob_data.bl, prob_data.bu),
    xtol=1e-8, gtol=1e-8, max_nfev=1000, verbose=2)
# x_opt, r_opt, g_opt, iterations = result
# final_obj = 0.5 * dot(r_opt, r_opt)
# converged = norm(g_opt, 2) < 1e-6 
prob_data = nls_functions[2]  # Use the first problem for testing
result = nonlinearlstr.lm_double_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
            )
result = nonlinearlstr.lm_interior_and_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result = nonlinearlstr.lm_trust_region_reflective_v2(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result = nonlinearlstr.active_set_svd_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result = nonlinearlstr.active_set_svd_trust_region_scaled(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result_tron = tron(eval(probs[2])())
result_tron
x_opt = result_tron.solution
final_obj = prob_data.obj_func(x_opt)
g_opt = prob_data.grad_func(x_opt)
iterations = result_tron.iter
converged = result_tron.status == :first_order


prob_data = nls_functions[end]  # Use the first problem for testing
result = nonlinearlstr.lm_double_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
            )
result = nonlinearlstr.lm_interior_and_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result = nonlinearlstr.lm_trust_region_reflective_v2(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)
result = nonlinearlstr.active_set_svd_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)

prob_data = create_nls_functions(eval(:tp242)())

result = nonlinearlstr.active_set_svd_trust_region(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)

result = nonlinearlstr.active_set_svd_trust_region_scaled(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
                max_iter=100, gtol=1e-8
)

result_tron = tron(eval(probs[2])())
result_tron
x_opt = result_tron.solution
final_obj = prob_data.obj_func(x_opt)
g_opt = prob_data.grad_func(x_opt)
iterations = result_tron.iter
converged = result_tron.status == :first_order
# result_tr = nonlinearlstr.bounded_trust_region(
#                 prob_data.residual_func, prob_data.jacobian_func, 
#                 prob_data.x0, prob_data.bl, prob_data.bu,;
#                 max_iter=100, gtol=1e-8
#             )

# result_fk = nonlinearlstr.fake_trust_region_reflective(
#                 prob_data.residual_func, prob_data.jacobian_func, 
#                 prob_data.x0, prob_data.bl, prob_data.bu,;
#                 max_iter=100, gtol=1e-8
#             )
# x_opt, r_opt, g_opt, iterations = result
# final_obj = 0.5 * dot(r_opt, r_opt)
# converged = norm(g_opt, 2) < 1e-6 

# --- Main Execution ---

# 1. Find and prepare problems
probs = find_bounded_problems()
lsresults = []
gtol = 1e-6

# 2. Loop through problems and run all solvers
for prob_name in probs
    println("--- Running Problem: $(prob_name) ---")
    
    # Create the NLPModels object for JSOSolvers
    prob_nlp = eval(prob_name)()
    # Create a separate struct with function handles for our solvers
    prob_data = create_nls_functions(prob_nlp)

    # --- Solver 1: lm_double_trust_region ---
    try
        result_dtr = nonlinearlstr.lm_double_trust_region(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_dtr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="Double-TR", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("  Double-TR failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="Double-TR", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 2: lm_interior_and_trust_region ---
    try
        result_itr = nonlinearlstr.lm_interior_and_trust_region(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_itr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="Interior-TR", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("  Interior-TR failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="Interior-TR", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 3: lm_trust_region_reflective ---
    try
        result_itr = nonlinearlstr.lm_trust_region_reflective(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_itr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="TRF", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("  TRF failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="TRF", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 4: lm_trust_region_reflective ---
    try
        result_itr = nonlinearlstr.lm_trust_region_reflective_v2(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_itr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="TRFv2", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("TRFv2 failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="TRFv2", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 5: active_set_svd_trust_region ---
    try
        result_itr = nonlinearlstr.active_set_svd_trust_region(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_itr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="active_set_svd_trust_region", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("active set solver failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="active_set_svd_trust_region", converged=false, final_obj=Inf, iterations=0))
    end

     # --- Solver 6: active_set_svd_trust_region_scaled ---
    try
        result_itr = nonlinearlstr.active_set_svd_trust_region_scaled(
            prob_data.residual_func, prob_data.jacobian_func, 
            prob_data.x0; lb=prob_data.bl, ub=prob_data.bu,
            max_iter=300, gtol=gtol
        )
        x_opt, r_opt, g_opt, iterations = result_itr
        final_obj = 0.5 * dot(r_opt, r_opt)
        converged = norm(g_opt, 2) < gtol
        push!(lsresults, (problem=prob_data.name, solver="active_set_svd_trust_region_scaled", converged=converged, final_obj=final_obj, iterations=iterations))
    catch e
        println("active set scaled solver failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="active_set_svd_trust_region_scaled", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 6: JSOSolvers.tron (Benchmark) ---
    try
        result_tron = tron(prob_nlp, atol=gtol, rtol=gtol)
        final_obj = result_tron.objective
        converged = result_tron.status == :first_order
        push!(lsresults, (problem=prob_data.name, solver="tron (JSO)", converged=converged, final_obj=final_obj, iterations=result_tron.iter))
    catch e
        println("tron (JSO) failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="tron (JSO)", converged=false, final_obj=Inf, iterations=0))
    end

    # --- Solver 7: SCIPY (Benchmark) ---
    try
        result_scipy = scipy.optimize.least_squares(prob_data.residual_func, prob_data.x0, jac=prob_data.jacobian_func, bounds=(prob_data.bl, prob_data.bu),
            xtol=1e-8, gtol=1e-8, max_nfev=1000, verbose=2)
        x = pyconvert(Vector{Float64}, result_scipy.x)
        final_obj = 0.5 * dot(prob_data.residual_func(x), prob_data.residual_func(x))
        converged = pyconvert(Bool, result_scipy.success) == true
        push!(lsresults, (problem=prob_data.name, solver="scipy", converged=converged, final_obj=final_obj, iterations=pyconvert(Int, result_scipy.nfev)))
    catch e
        println("  SCIPY failed: $e")
        push!(lsresults, (problem=prob_data.name, solver="scipy", converged=false, final_obj=Inf, iterations=0))
    end
    # Finalize the NLPModels object to free resources
    finalize(prob_nlp)
end

# 3. Create a DataFrame and analyze results with Tidier
df = DataFrame(lsresults)

# Find the best objective for each problem to calculate a performance ratio
df_best = @chain df begin
    @group_by(problem)
    @summarize(best_obj = minimum(final_obj))
end

# Join the best objective back to the main dataframe
df_perf = @chain df begin
    @left_join(df_best, problem)
    @mutate(practical_convergence = abs.(final_obj .- best_obj) .< 1e-6)
end

println("\n\n" * "="^60)
println("SOLVER PERFORMANCE SUMMARY")
println("="^60)

# Use Tidier to create a summary table
summary_table = @chain df_perf begin
    @group_by(solver)
    @summarize(
        n_problems = n(),
        n_converged = sum(practical_convergence),
        median_iters = median(iterations),
    )
    @mutate(
        success_rate = round(100 * n_converged / n_problems, digits = 2)
    )
    @select(solver, success_rate, n_converged, median_iters)
end

# Print the summary table
println(summary_table)
println("\n'median_perf_ratio' is the median of (final_obj / best_obj_for_this_problem). Lower is better (1.0 is best).")

# For the solvers with more than 70% performance, show which problems they failed 
# and how far they were from the best objective
println("\n" * "="^60)
println("DETAILED FAILURE ANALYSIS FOR HIGH-PERFORMING SOLVERS")
println("="^60)

for row in eachrow(summary_table)
    if row.success_rate >= 70  # Changed from < 70 to >= 70
        println("\nSolver '$(row.solver)' had a success rate of $(row.success_rate)%. Failed problems:")
        
        # Use regular DataFrames filtering instead of Tidier inside the loop
        failed_problems = df_perf[
            (df_perf.solver .== row.solver) .& (.!df_perf.practical_convergence), 
            [:problem, :final_obj, :best_obj]
        ]
        
        # Add the ratio and difference columns
        failed_problems.ratio = failed_problems.final_obj ./ failed_problems.best_obj
        failed_problems.obj_diff = failed_problems.final_obj .- failed_problems.best_obj
        
        # Sort by worst performance first
        sort!(failed_problems, :ratio, rev=true)
        
        if nrow(failed_problems) == 0
            println("  → No failed problems! Perfect solver performance.")
        else
            println("  → $(nrow(failed_problems)) failed problems:")
            for (i, fail_row) in enumerate(eachrow(failed_problems))
                if isfinite(fail_row.ratio)
                    println("    $(i). $(fail_row.problem): final_obj=$(round(fail_row.final_obj, digits=6)), " *
                           "best_obj=$(round(fail_row.best_obj, digits=6)), " *
                           "ratio=$(round(fail_row.ratio, digits=3))x worse")
                else
                    println("    $(i). $(fail_row.problem): DIVERGED (final_obj=Inf)")
                end
            end
        end
    end
end

# Additional analysis: Show which problems are hardest (failed by most solvers)
println("\n" * "="^60)
println("MOST CHALLENGING PROBLEMS (failed by multiple solvers)")
println("="^60)

# Use regular DataFrames groupby instead of Tidier
problem_difficulty = combine(
    groupby(df_perf, :problem),
    :practical_convergence => (x -> length(x) - sum(x)) => :n_solvers_failed,
    :practical_convergence => length => :n_total_solvers,
    :best_obj => first => :best_obj
)

# Add failure rate and filter
problem_difficulty.failure_rate = round.(100 * problem_difficulty.n_solvers_failed ./ problem_difficulty.n_total_solvers, digits=1)
problem_difficulty = problem_difficulty[problem_difficulty.n_solvers_failed .>= 2, :]
sort!(problem_difficulty, :failure_rate, rev=true)

if nrow(problem_difficulty) > 0
    println("Problems that failed for multiple solvers:")
    for (i, prob_row) in enumerate(eachrow(problem_difficulty))
        println("  $(i). $(prob_row.problem): $(prob_row.n_solvers_failed)/$(prob_row.n_total_solvers) solvers failed " *
               "($(prob_row.failure_rate)% failure rate)")
    end
else
    println("No problems failed for multiple solvers - good solver robustness!")
end
