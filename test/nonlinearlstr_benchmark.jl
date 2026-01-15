include("nlls_problems_prep.jl")
using DataFrames
using nonlinearlstr
using Test
using CairoMakie

# 1. Find problems
nls_problems = find_nlls_problems(20) # Benchmark on all available NLS problems

# 2. Define solvers
solvers = [
    ("LM-SVD", nonlinearlstr.lm_trust_region),
    ("LM-QR", nonlinearlstr.lm_trust_region),
    ("LM-QR-Recursive", nonlinearlstr.lm_trust_region),
]

# 3. Run benchmark
println("Running benchmark on $(length(nls_problems)) problems...")
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)

# 4. Analyze results
df_nls = DataFrame(nls_results)

include("evaluate_solver_dfs.jl")

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)

println("\n\n" * "="^60)
println("NONLINEARLSTR SOLVER PERFORMANCE SUMMARY")
println("="^60)
display(summary_nls)

# 5. Plots
fig_nls = build_performance_plots(df_nls_proc)
if !isdir("test_plots")
    mkdir("test_plots")
end
save("test_plots/nonlinearlstr_solver_performance.png", fig_nls)
println("\nPerformance plot saved to 'test_plots/nonlinearlstr_solver_performance.png'")
