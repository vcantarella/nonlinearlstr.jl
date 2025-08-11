using CUTEst
using NLPModels
using JSOSolvers
using PRIMA
using NonlinearSolve
using Enlsip
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
        ("QR-NLLS", nonlinearlstr.qr_nlss_trust_region),
        ("LM-TR", nonlinearlstr.lm_trust_region),
        ("LM-TR-scaled", nonlinearlstr.lm_trust_region_scaled),
        ("PRIMA-NEWUOA", nothing),  # Special handling
        ("PRIMA-BOBYQA", nothing),  # Special handling
        ("NL-TrustRegion", NonlinearSolve.TrustRegion),  # Special handling
        ("NL-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),  # Special handling
        ("NL-GaussNewton", NonlinearSolve.GaussNewton),  # Special handling
        ("NL-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),  # Special handling
        ("TRON", tron),  # Special handling
        ("TRUNK", trunk),  # Special handling
        ("LSO-DogLeg-QR", LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR())),
        ("LSO-Levenberg-QR", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.QR())),
        ("LSO-DogLeg-chol", LeastSquaresOptim.Dogleg(LeastSquaresOptim.Cholesky())),
        ("LSO-Levenberg-chol", LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.Cholesky())),
        #("Enlsip", nothing)
    ]

nls_results = nlls_benchmark(nls_problems, solvers, max_iter=400)
#cutest_results = nlls_benchmark(cutest_problems, solvers, max_iter=100)

# Convert to DataFrame
df_nls = DataFrame(nls_results)

df_nls_proc = @chain df_nls begin
    @group_by(problem)
    @mutate(min_solution = max(0, minimum(final_objective)))
    @ungroup
    @mutate(final_close = ifelse((abs(final_objective - min_solution) <= 1e-4) & (final_objective >= 0.0),
        true, false))
end

# group by solver and
@chain df_nls_proc begin
    @group_by(solver)
    @summarize(
        converged = sum(final_close)/maximum(final_close),
        iterations = median(iterations),
        mean_objective = mean(final_objective),
        std_objective = std(final_objective)
    )
    @arrange(desc(converged))
end

#Check and return the problems where the performance with QR-NLLS is bad
df_nls_bad = @chain df_nls_proc begin
    @filter(solver == "LM-TR")
    @filter(final_close == false)
end

bad_probs = df_nls_bad.problem
#convert to symbols
bad_probs_symb = Symbol.(bad_probs)
df_bad_probs = DataFrame(nlls_benchmark(bad_probs_symb, solvers, max_iter=200))

first_bad_prob = bad_probs[3]
df_bad_probs[df_bad_probs.problem .== first_bad_prob, :]


df_mgh33 = @chain df_nls_proc begin
    @filter(problem == "tp202")
end

bad_probs = df_nls_bad.problem[df_nls_bad.converged .== false]
#convert to symbols
bad_probs_symb = Symbol.(bad_probs)
df_bad_probs = DataFrame(nlls_benchmark(bad_probs_symb, solvers, max_iter=200))

first_bad_prob = bad_probs[5]
df_bad_probs[df_bad_probs.problem .== first_bad_prob, :]
# Convert to DataFrame
#df_cutest = DataFrame(cutest_results)no