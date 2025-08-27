using CSV, DataFrames
using NonlinearSolve
using nonlinearlstr
using ForwardDiff

func(x, p) = 1 ./ (p[1]+p[2] * exp.(-p[3] * x ))
F_averaged_vi = [-3.15, -2.25, -1.95, -1.65, -1.35, -1.05, -0.75, -0.45, -0.15, 0.15, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.55]
F_VarY = [0.1, 0.65, 0.88, 0.883333, 1.94286, 1.28, 0.766667, 1.36071, 1.65405, 2.425, 3.23333, 3.80286, 6.23333, 8.1375, 21.9, 1.3, 24.6]

println(F_averaged_vi)
println(F_VarY)


p0 = [0.01, 0.5, 0.5]
func.(F_averaged_vi, Ref(p0))
residual_func(p) = F_VarY - func.(F_averaged_vi, Ref(p))
jac_func(p) = ForwardDiff.jacobian(residual_func, p)

res = nonlinearlstr.lm_svd_trust_region(
    residual_func, jac_func, p0;
    max_iter = 100, gtol = 1e-8
)
cost = 0.5 * residual_func(res[1])'* residual_func(res[1])
res_nl(u, p) = residual_func(u)
prob = NonlinearLeastSquaresProblem(res_nl, p0)
sol = solve(prob;
    show_trace = Val(true),
    trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(1)
    )
costnl = 1/2 * residual_func(sol.u)'*residual_func(sol.u)

using BenchmarkTools

@benchmark nonlinearlstr.lm_svd_trust_region(
    $residual_func, $jac_func, $p0;
    max_iter = 100, gtol = 1e-8
)

@benchmark solve($prob;)
@benchmark solve($prob, TrustRegion();)
@benchmark solve($prob, LevenbergMarquardt();)
