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
using NLSProblems

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

nlp = eval(:NZF1)()
x0 = copy(nlp.meta.x0)
bl = copy(nlp.meta.lvar)
bu = copy(nlp.meta.uvar)
# For CUTEst NLLS problems with objtype="none", the residuals are the constraints
residual_func(x) = residual(nlp, x)
jacobian_func(x) = Matrix(jac_residual(nlp, x))
n,m = size(jacobian_func(x0))
# Create objective as 0.5 * ||r||²
obj_func(x) = 0.5 * dot(residual_func(x), residual_func(x))
grad_func(x) = jacobian_func(x)' * residual_func(x)

# Use Gauss-Newton approximation for Hessian
hess_func(x) = begin
    J = jacobian_func(x)
    return J' * J
end

res = nonlinearlstr.qr_nlss_trust_region(
    residual_func, jacobian_func, x0;
    max_iter = 100, gtol = 1e-8,
)

res2 = nonlinearlstr.lm_trust_region(
    residual_func, jacobian_func, x0;
    max_iter = 100, gtol = 1e-8,
)

res3 = nonlinearlstr.lm_trust_region_v2(
    residual_func, jacobian_func, x0;
    max_iter = 100, gtol = 1e-8,
)
res4 = nonlinearlstr.lm_fan_lu(
    residual_func, jacobian_func, x0;
    max_iter = 100, gtol = 1e-8,
)

nlp = eval(:tp210)()
x0 = copy(nlp.meta.x0)
bl = copy(nlp.meta.lvar)
bu = copy(nlp.meta.uvar)
# For CUTEst NLLS problems with objtype="none", the residuals are the constraints
residual_func(x) = residual(nlp, x)
jacobian_func(x) = Matrix(jac_residual(nlp, x))
n,m = size(jacobian_func(x0))
# Create objective as 0.5 * ||r||²
obj_func(x) = 0.5 * dot(residual_func(x), residual_func(x))
grad_func(x) = jacobian_func(x)' * residual_func(x)

# Use Gauss-Newton approximation for Hessian
hess_func(x) = begin
    J = jacobian_func(x)
    return J' * J
end

res = nonlinearlstr.qr_nlss_trust_region(
    residual_func, jacobian_func, x0;
    max_iter = 500, gtol = 1e-8
)
res2 = nonlinearlstr.lm_trust_region(
    residual_func, jacobian_func, x0;
    max_iter = 500, gtol = 1e-8
)
grad_func(res[1])
n_res(u, p) = residual_func(u)
nl_jac(u, p) = jacobian_func(u)
nl_func = NonlinearFunction(n_res, jac=nl_jac)
prob_nl = NonlinearLeastSquaresProblem(nl_func, x0)
sol = solve(prob_nl, NonlinearSolve.SimpleNewtonRaphson(); maxiters=400,
    show_trace = Val(true),
    trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(1)
)
grad_func(sol.u)