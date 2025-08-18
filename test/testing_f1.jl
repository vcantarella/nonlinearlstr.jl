using NonlinearSolve
using ForwardDiff
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LinearAlgebra, Statistics

using nonlinearlstr

f_ex3(t::AbstractArray, a) = @. a[1] * exp.(-t / a[2]) + a[3] * sin.(t / a[4])
t = 0.0:1:100  # Time points
a_true = [  6.0 , 20.0 , 1.0 , 5.0 ]
a0 = [ 10.0 , 50.0 ,  5.0 ,  5.7 ]
y_clean = f_ex3(t, a_true)
#noise_level = 0.05  # Noise level
#noise = noise_level * randn(length(y_clean)) * mean(y_clean)
y_data = y_clean #+ noise
residual_func(a) = y_data-f_ex3(t,a)
jac_func(a) = ForwardDiff.jacobian(residual_func, a)
lb = repeat([-Inf], inner=4)
ub = repeat([Inf], inner=4)
f_true = residual_func(a_true)
cost_true = 0.5 * dot(f_true, f_true)  # True cost value

# Running the lm_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region(
    residual_func, jac_func, a0;
    max_iter = 100, gtol = 1e-8, shrink_factor = 0.5
)

# Running the lm_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region_v2(
    residual_func, jac_func, a0;
    max_iter = 100, gtol = 1e-8, shrink_factor = 0.5
)

a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_fan_lu(
    residual_func, jac_func, a0;
    max_iter = 100, gtol = 1e-8, shrink_factor = 0.5
)

a_opt, f_opt, g_opt, iter = nonlinearlstr.svd_trust_region(
    residual_func, jac_func, a0;
    max_iter = 100, gtol = 1e-8, shrink_factor = 0.5
)


# Running the lm_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region_scaled(
    residual_func, jac_func, a0;
    max_iter = 100, gtol = 1e-8, 
)


res_nl(u,p) = residual_func(u)
nlprob = NonlinearLeastSquaresProblem(res_nl, a0)
sol = solve(nlprob, TrustRegion();
    show_trace = Val(true),
    trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(1)
    )
a_opt = sol.u