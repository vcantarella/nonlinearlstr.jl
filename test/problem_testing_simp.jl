include("nlls_problems_prep.jl")
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Revise
using DataFrames
using nonlinearlstr

nls_problems = find_nlls_problems(999)

solvers = [
    # nonlinearlstr solvers (keep all)
    ("LM-QR", nonlinearlstr.lm_trust_region),
    ("LM-SVD", nonlinearlstr.lm_trust_region),
    
    # PRIMA (Best: NEWUOA for unconstrained)
    ("PRIMA-NEWUOA", nothing), 

    # NonlinearSolve.jl (keep all)
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion), 
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton), 
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),
    
    # JSOSolvers (Best: TRON)
    ("JSO-TRON", tron),

    # LeastSquaresOptim (Best: Levenberg-QR)
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),

    # Scipy (Best: LeastSquares)
    ("Scipy-LeastSquares", nothing),

    # NLLSsolver (Best: LevenbergMarquardt)
    ("NLLSsolver-levenbergmarquardt", NLLSsolver.levenbergmarquardt),
]

# Run benchmark
nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)

# Convert to DataFrame
df_nls = DataFrame(nls_results)

include("evaluate_solver_dfs.jl")

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)
display(summary_nls)

using Test
@testset "Solver Performance Tests" begin
    # Check that our solvers perform reasonably well (success rate > 90% relative to best)
    # Note: These thresholds might need adjustment based on the specific problem set difficulty
    if !isempty(summary_nls)
        qr_row = summary_nls[summary_nls.solver .== "LM-QR", :]
        svd_row = summary_nls[summary_nls.solver .== "LM-SVD", :]
        
        if !isempty(qr_row)
            @test qr_row[1, :percentage_success] > 0.9
        end
        if !isempty(svd_row)
            @test svd_row[1, :percentage_success] > 0.9
        end
    end
end

fig_nls = build_performance_plots(df_nls_proc)
if !isdir("../test_plots")
    mkdir("../test_plots")
end
save("../test_plots/nlls_solver_performance.png", fig_nls)
