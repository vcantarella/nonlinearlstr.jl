using OrdinaryDiffEq
using SciMLSensitivity
using ForwardDiff
using LinearAlgebra
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.5:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5(); saveat = tsteps)

# Generate error-prone heterosckedastic measurements
using Random
Random.seed!(6674823)  # Set seed for reproducibility

u = sol.u
u1 = [ux[1] for ux in u]
σ1 = 0.1 * u1
u2 = [ux[2] for ux in u]
σ2 = 0.1 * u2
u_noisy1 = u1 + σ1 .* randn(size(u))
u_noisy2 = u2 + σ2 .* randn(size(u))

# Define the cost function
function cost(p)
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, Rosenbrock23(); saveat = tsteps, reltol = 1e-9, abstol = 1e-9)
    u = sol.u
    u1 = [ux[1] for ux in u]
    u2 = [ux[2] for ux in u]
    cost1 = sum((u_noisy1 .- u1).^2 ./ σ1.^2)
    cost2 = sum((u_noisy2 .- u2).^2 ./ σ2.^2)
    return cost1 + cost2
end

# Define the gradient of the cost function
function grad_cost(p)
    return ForwardDiff.gradient(cost, p)
end

cost(p)
grad_cost(p)
hess(p) = ForwardDiff.hessian(cost, p)
# hess(p) = grad_cost(p)' * grad_cost(p)

# Define the bounds
lb = [0.0, 0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
# sample from an uniform distribution within the bounds
x0 = [rand() * (ub[i] - lb[i]) + lb[i] for i in 1:4]
cost(x0)
# Run the optimization
using nonlinearlstr
# opt = nonlinearlstr.bounded_trust_region(cost, grad_cost, hess,x0, lb, ub;step_threshold = 1e-4,
#     initial_radius = 1e0,
#  max_iter = 1000, gtol = 1e-8, min_trust_radius = 1e-9, max_trust_radius = 100)
# x = opt[1]
# @show cost(x)

function resi(x)
    problem = ODEProblem(lotka_volterra!, u0, tspan, x)
    sol = solve(problem, Rosenbrock23(); saveat = tsteps, reltol = 1e-9, abstol = 1e-9)
    u = sol.u
    u1 = [ux[1] for ux in u]
    u2 = [ux[2] for ux in u]
    [(u_noisy1 .- u1)./σ1; (u_noisy2 .- u2)./σ2]
    
end

function jac(x)
    return ForwardDiff.jacobian(resi, x)
end

using PRIMA
res = PRIMA.bobyqa(cost, x0, xl=lb, xu=ub, iprint = PRIMA.MSG_EXIT)
x_p = res[1]
@show cost(x_p)



using NonlinearSolve
res_2(x, p) = resi(x)
nlls_prob = NonlinearLeastSquaresProblem(res_2, x0)
nlls_sol = solve(nlls_prob, TrustRegion();
    maxiters = 100, show_trace = Val(true),
    trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(25))
p_nl = nlls_sol.u
@show cost(p_nl)


# opt_nls = nonlinearlstr.nlss_bounded_trust_region(resi, jac, x0, lb, ub;step_threshold = 1e-5,
#     initial_radius = 1e0,
#  max_iter = 1000, gtol = 1e-15, min_trust_radius = 1e-12, max_trust_radius = 1000)

# Test the new QR-based solver
opt_qr = nonlinearlstr.qr_nlss_bounded_trust_region(resi, jac, x0, repeat([-Inf], length(x0)), repeat([Inf], length(x0));
    initial_radius = 1e0,
    max_iter = 100, gtol = 1e-8, min_trust_radius = 1e-8, max_trust_radius = 1000)

# x = opt_nls[1]
# @show cost(x)
x_qr = opt_qr[1]
@show cost(x_qr)
println("QR solver vs old solver cost ratio: ", cost(x_qr) / cost(x0))
function hess_approx(x) 
    J = jac(x)
    QR = qr(J)
    return 4 * QR.R' * QR.R
end
opt_v2 = nonlinearlstr.bounded_trust_region(cost, grad_cost,hess_approx ,x0, repeat([-Inf], length(x0)), repeat([Inf], length(x0));
    step_threshold = 1e-5,
    initial_radius = 1e0,
 max_iter = 100, gtol = 1e-8, min_trust_radius = 1e-6, max_trust_radius = 100)
x_v2 = opt_v2[1]
@show cost(x_v2)
