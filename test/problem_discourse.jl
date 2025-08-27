using CSV, DataFrames
using NonlinearSolve
using nonlinearlstr
using ForwardDiff

df = CSV.read("test/discourse_data.csv", DataFrame)

data = [df.C_EA df.C_EC df.q_EA_303K df.q_EC_303K]

function loss_function(p, data)
    Ha, b_EA, b_EC, Hc = p
    res1 = @. Ha * data[:, 1] / (1 + b_EA * data[:, 1] + b_EC * data[:, 2]) - data[:, 3] # data[:, 1] = C_EA, data[:, 2] = C_EC
    res2 = @. Hc * data[:, 2] / (1 + b_EA * data[:, 1] + b_EC * data[:, 2]) - data[:, 4] # data[:, 3] = q_EA, data[:, 4] = q_EC,
    return vcat(res1, res2)
end

p_init = [1.983, 0.0258, 0.2037, 1.53]

loss_function(p_init, data)  # Initial residuals

nlls_prob = NonlinearLeastSquaresProblem(loss_function, p_init, data)

res = solve(nlls_prob, LevenbergMarquardt(); maxiters = 1000, show_trace = Val(true),
trace_level = TraceWithJacobianConditionNumber(25))

loss_f(data) = p -> loss_function(p, data)
jac_f(data) = p -> ForwardDiff.jacobian(loss_f(data), p)

a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region(
    loss_f(data), jac_f(data), p_init;
    max_iter = 100, gtol = 1e-8, shrink_factor = 0.5
)
a_opt
res.u

a_opt ≈ res.u
