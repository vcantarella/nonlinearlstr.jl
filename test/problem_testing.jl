using CUTEst
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LinearAlgebra, Statistics
using Tidier

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr
include("nlls_problems_prep.jl")

nls_problems = find_nlls_problems(999)
#cutest_problems = find_cutest_nlls_problems(999)


solvers = [
        ("QR-NLLS", nonlinearlstr.qr_nlss_bounded_trust_region),
        ("QR-NLLS-scaled", nonlinearlstr.qr_nlss_bounded_trust_region_v2),
        ("PRIMA-NEWUOA", nothing),  # Special handling
        ("PRIMA-BOBYQA", nothing),  # Special handling
        ("NL-TrustRegion", NonlinearSolve.TrustRegion),  # Special handling
        ("NL-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),  # Special handling
        ("NL-GaussNewton", NonlinearSolve.GaussNewton),  # Special handling
        ("TRON", tron),  # Special handling
        ("TRUNK", trunk),  # Special handling
        ("LSO-DogLeg-QR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR())),
        ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
        ("LSO-DogLeg-chol", LeastSquaresOptim.Dogleg(LeastSquaresOptim.Cholesky())),
        ("LSO-Levenberg-chol", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky())),
    ]

nls_results = nlls_benchmark(nls_problems, solvers, max_iter=100)
cutest_results = nlls_benchmark(cutest_problems, solvers, max_iter=100)

# Convert to DataFrame
df_nls = DataFrame(nls_results)
# Convert to DataFrame
df_cutest = DataFrame(cutest_results)