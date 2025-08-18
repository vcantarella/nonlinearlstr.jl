using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using ForwardDiff
using LinearAlgebra, Statistics

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

# Hard Problems in Luksan, 1995
fa1(x,t) = x[1] + x[2]*exp(x[3]*t)
ti1 = [1,5,10,15,20,25,30,35,40,50]
yi1 = [16.7, 26.8, 16.9, 17.1, 17.2, 17.4, 17.6, 17.9, 18.1, 18.7]
resa1(x) = [fa1(x, t) - y for (t, y) in zip(ti1, yi1)]
x01 = [20, 2, 0.5]
resa1(x01)
fa2(x,t) = exp(x[1]*t)+exp(x[2]*t)
ti2 = [1,2,3,4,5,6,7,8,9,10]
yi2 = [4,6,8,10,12,14,16,18,20,22]
resa2(x) = [fa2(x, t) - y for (t, y) in zip(ti2, yi2)]
x02 = [0.3, 0.2]
resa2(x02)
fa3(x,t) = x[1]*exp(x[2]/(x[3]+t))
ti3 = [50, 55, 60, 65, 70, 75, 80, 85, 90,
 95, 100, 105, 110, 115,120, 125]
yi3 = [34780, 28610, 23650, 19630, 16370, 13720, 11540,
    9744, 8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872]
resa3(x) = [fa3(x, t) - y for (t, y) in zip(ti3, yi3)]
x03 = [0.02, 4000, 250]
resa3(x03)
fa4(x,t) = x[1]*exp(-x[3]*t)+x[2]*exp(-x[4]*t)
ti4 = [1, 2, 3, 4, 5, 6,7 ,8,9,10]
yi4 = [99.6, 67.1, 45.9, 31.9, 22.5, 16.1, 11.7, 8.6, 6.38, 4.78]
resa4(x) = [fa4(x, t) - y for (t, y) in zip(ti4, yi4)]
x04 = [1, 1, 1, 1]
fa5(x,t) = x[1]*exp(-x[3]*t)+x[2]*exp(-x[4]*t)
ti5 = [7.448, 7.448, 7.552, 7.607, 7.847, 7.877,7.969,8.176,
    8.176, 8.523, 8.552, 8.903, 9.114, 9.284, 9.439]
yi5 = [57.554, 53.546, 45.29, 51.286, 31.623, 27.952, 19.498,
    16.444, 21.777, 13.996, 11.803, 7.727, 4.764, 4.305, 3.006]
resa5(x) = [fa5(x, t) - y for (t, y) in zip(ti5, yi5)]
x05 = [1e5, 1e5, 1.079, 1.31]
resa5(x05)
fa6(x,t) = x[1]*t^x[3] + x[2]*t^x[4]
ti6 = 12:23
yi6 = [7.31, 7.55, 7.8, 8.05, 8.31, 8.57, 8.84, 9.12, 9.4, 9.69, 9.99, 10.3]
resa6(x) = [fa6(x, t) - y for (t, y) in zip(ti6, yi6)]
x06 = [1e3, 0.01, 2, 100]
resa6(x06)

function build_problems(resf, x0)
    jac = x -> ForwardDiff.jacobian(resf, x)
    prob_data = (
        res = resf,
        jac = jac,
        grad = x-> jac(x)'resf(x),
        obj = x -> 0.5 * dot(resf(x), resf(x)),
        x0 = x0,
        initial_obj = 0.5 * dot(resf(x0), resf(x0))
    )
    return prob_data
end

solvers = solvers = [
        ("QR-NLLS", nonlinearlstr.qr_nlss_trust_region),
        ("LM-TR", nonlinearlstr.lm_trust_region),
        ("LM-TR-scaled", nonlinearlstr.lm_trust_region_scaled),
        ("PRIMA-NEWUOA", PRIMA.newuoa),  # Special handling
        ("PRIMA-BOBYQA", PRIMA.bobyqa),  # Special handling
        ("NL-TrustRegion", NonlinearSolve.TrustRegion),  # Special handling
        ("NL-LevenbergMarquardt", NonlinearSolve.LevenbergMarquardt),  # Special handling
        ("NL-GaussNewton", NonlinearSolve.GaussNewton),  # Special handling
        ("NL-PolyAlg", NonlinearSolve.FastShortcutNLLSPolyalg),  # Special handling
    ]
function solve_non(solver_func, prob_data; maxiters)
    residual_func = prob_data.res
    jac_func = prob_data.jac
    grad_func = prob_data.grad
    obj_func = prob_data.obj
    x0 = prob_data.x0
    initial_obj = prob_data.initial_obj
    sol = solver_func(
        residual_func, jac_func, x0, ;
        max_iter=maxiters    )
    x, f_opt, g_opt, iter = sol
    # Basic convergence tests
    converged = norm(g_opt, 2) < 1e-6
    cost = 0.5 * dot(f_opt, f_opt)

    return (
        x0 = x0,
        x_opt = x,
        converged = converged,
        iterations = iter,
        initial_cost = initial_obj,
        cost = cost,
        )
end
function solve_nlsolve(solver_func, prob_data; maxiters)
    residual_func = prob_data.res
    nl_res(u,p) = residual_func(u)
    jac_func = prob_data.jac
    x0 = prob_data.x0
    initial_obj = prob_data.initial_obj

    nlprob = NonlinearLeastSquaresProblem(nl_res, x0)
    sol = solve(nlprob, solver_func(); maxiters=maxiters)
    x_opt = sol.u
    f_opt = residual_func(x_opt)
    g_opt = jac_func(x_opt)' * f_opt  # Gradient of 0.
    iter = sol.stats.nsteps
    # Basic convergence tests
    converged = norm(g_opt, 2) < 1e-6
    cost = 0.5 * dot(f_opt, f_opt)
    return (
        x0 = x0,
        x_opt = x_opt,
        converged = converged,
        iterations = iter,
        initial_cost = initial_obj,
        cost = cost,
        )
end
function solve_prima(solver_func, prob_data; maxiters)
    residual_func = prob_data.res
    obj = prob_data.obj
    jac_func = prob_data.jac
    x0 = prob_data.x0
    initial_obj = prob_data.initial_obj
    res = solver_func(obj, x0)
    x_opt = res[1]
    f_opt = residual_func(x_opt)
    g_opt = jac_func(x_opt)' * f_opt  # Gradient of 0.
    iter = res[2].nf
    # Basic convergence tests
    converged = PRIMA.issuccess(res[2])
    cost = 0.5 * dot(f_opt, f_opt)
    return (
        x0 = x0,
        x_opt = x_opt,
        converged = converged,
        iterations = iter,
        initial_cost = initial_obj,
        cost = cost,
        )
end

problems = [ build_problems(res,x0) for (res,x0) in zip(
    [resa1, resa2, resa3, resa4, resa5, resa6],
    [x01, x02, x03, x04, x05, x06]
)]
problem_names = ["A.1", "A.2", "A.3", "A.4", "A.5", "A.6"]
results = []
for (name, prob) in zip(problem_names, problems)
    println("Problem Name: $name")
    println("Problem Data: $prob")
    for (solver_name, solver_func) in solvers
        println("  Solver Name: $solver_name")
        println("  Solver Function: $solver_func")
        if solver_name in ["QR-NLLS", "LM-TR", "LM-TR-scaled"]
            result = solve_non(solver_func, prob; maxiters=100)
        elseif solver_name in ["NL-TrustRegion", "NL-LevenbergMarquardt", "NL-GaussNewton", "NL-PolyAlg"]
            result = solve_nlsolve(solver_func, prob; maxiters=100)
        elseif solver_name in ["PRIMA-NEWUOA", "PRIMA-BOBYQA"]
            result = solve_prima(solver_func, prob; maxiters=100)
        end
        result = merge(result, (prob_name=name, solver=solver_name))
        push!(results, result)
    end
end

df = DataFrame(results)

name = "A.1"
prob = problems[4]
println("Problem Name: $name")
println("Problem Data: $prob")
solver_name = "LM-TR"
solver_func= nonlinearlstr.lm_trust_region
println("  Solver Name: $solver_name")
println("  Solver Function: $solver_func")
result = solve_non(solver_func, prob; maxiters=200)