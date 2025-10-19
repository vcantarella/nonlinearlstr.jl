include("nlls_problems_prep.jl")
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames
using nonlinearlstr

nls_problems = find_nlls_problems(999)


solvers = [
    ("LM-QR", nonlinearlstr.lm_trust_region),
    ("LM-QR-scaled", nonlinearlstr.lm_trust_region),
    ("LM-SVD", nonlinearlstr.lm_trust_region),
    ("PRIMA-NEWUOA", nothing),  # Special handling
    ("PRIMA-BOBYQA", nothing),  # Special handling
    ("NonlinearSolve-TrustRegion", NonlinearSolve.TrustRegion),  # Special handling
    ("NonlinearSolve-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),  # Special handling
    ("NonlinearSolve-GaussNewton", NonlinearSolve.GaussNewton),  # Special handling
    ("NonlinearSolve-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),  # Special handling
    ("JSO-TRON", tron),  # Special handling
    ("JSO-TRUNK", trunk),  # Special handling
    ("LSO-DogLeg-QR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR())),
    ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
    ("LSO-DogLeg-chol", LeastSquaresOptim.Dogleg(LeastSquaresOptim.Cholesky())),
    (
        "LSO-Levenberg-chol",
        LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky()),
    ),
    ("Scipy-LeastSquares", nothing),  # Special handling
]

nls_results = nlls_benchmark(nls_problems, solvers, max_iter = 400)
#cutest_results = nlls_benchmark(cutest_problems, solvers, max_iter=100)

# Convert to DataFrame
df_nls = DataFrame(nls_results)

include("evaluate_solver_dfs.jl")

df_nls_proc = compare_with_best(df_nls)
summary_nls = evaluate_solvers(df_nls_proc)
display(summary_nls)

@test summary_df[summary_df[:solver] .== "LM-QR", :percentage_success][1] > 0.9
@test summary_df[summary_df[:solver] .== "LM-SVD", :percentage_success][1] > 0.9
fig_nls = build_performance_plots(df_nls_proc)
save("../test_plots/nlls_solver_performance.png", fig_nls)

# # testing scipy for debugging
# prob_name = nls_problems[3]
# nlp = eval(prob_name)()
# prob_data = create_nls_functions(nlp)
# test_solver_on_problem("Scipy-LeastSquares", nothing, prob_data, nlp)
# scipy.optimize.least_squares(prob_data.residual_func, prob_data.x0, jac=prob_data.jacobian_func, bounds=(prob_data.bl, prob_data.bu),
#             xtol=1e-8, gtol=1e-8, max_nfev=1000, verbose=2)
