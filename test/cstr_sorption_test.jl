# I want to test different ODE formulations for the equilibrium sorption model.
using OrdinaryDiffEq
using NonlinearSolve
using CairoMakie
using LinearAlgebra
using Random
using nonlinearlstr
using ForwardDiff
using PRIMA
using Optimization
"""
    f_eq(c1, c2, smax, K1, K2)
Calculate the equilibrium sorbed concentrations for two components
     based on their concentrations and equilibrium constants using the Langmuir competitive sorption model.
"""
function f_eq(c1, c2 ,smax, K1, K2)
    s1_eq = smax * c1 / (K1*(1+c2/K2) + c1)
    s2_eq = smax * c2 / (K2*(1+c1/K1) + c2)
    return s1_eq, s2_eq
end

# CSTR model for equilibrium sorption assuming we need to equilibrate the sorb and liquid concentrations
"""
    rhs_eq_sorption_v2!(du, u, p, t)
Calculate the right-hand side of the ODE for the equilibrium sorption model.
Assuming we DON'T need to equilibrate the sorb and liquid concentrations to calculate the rates of exchange.
"""
function make_sorption_model(c_in1, c_in2, V, Q, ρₛ, ϕ)
function rhs_eq_sorption_v2!(du, u, p, t)
    smax, K1, K2, λ = p
    c1 = u[1]
    c2 = u[2]
    s1 = u[3]
    s2 = u[4]
    s_eq = f_eq(c1, c2, smax, K1, K2)
    du[1] = (Q/(V*ϕ)) * (c_in1 - c1) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[1] - s1)
    du[2] = (Q/(V*ϕ)) * (c_in2 - c2) - (1-ϕ)/ϕ * ρₛ * λ * (s_eq[2] - s2)
    du[3] = λ * (s_eq[1] - s1)
    du[4] = λ * (s_eq[2] - s2)
end
return rhs_eq_sorption_v2!
end

u0 = [0, 0, 0, 0] # Initial concentrations
c_in1 = 2e-3 # Inlet concentration
c_in2 = 1e-3 # Inlet concentration
V = 0.1 # l Volume of the reactor
Q = 1e-4 # Flow rate in l/s
ρₛ = 2.65 # Density of the sorbent [kg/l]
ϕ = 0.3 # Porosity
rhs = make_sorption_model(c_in1, c_in2, V, Q, ρₛ, ϕ)
smax = 0.0034 # Maximum sorbed concentration
K1 = 1e-3 # Equilibrium constant
K2 = 2e-4 # Equilibrium constant
λ = 0.1e-2 # Rate constant for sorption
p = [smax, K1, K2, λ]
du0 = zeros(eltype(u0), length(u0))
# detector = TracerSparsityDetector()
# jac_sparsity2 = ADTypes.jacobian_sparsity((du, u) -> rhs_eq_sorption!(du, u, p, 1),
#     du0, u0, detector) # add the sparsity pattern to speed up the solution

tspan = (0.0, 6000.0) # Time span for the simulation
# Shut the system down with a callback
prob_v2 = ODEProblem(rhs, u0, tspan, p)
sol_v2 = solve(prob_v2, Tsit5(), abstol=1e-8, reltol=1e-8, saveat= 200)
# Generate error-prone heterosckedastic measurements
Random.seed!(6674823)  # Set seed for reproducibility
u = sol_v2.u
u1 = [ux[1] for ux in u]
σ1 = 0.05 * u1
u2 = [ux[2] for ux in u]
σ2 = 0.05 * u2
u_noisy1 = u1 + σ1 .* randn(size(u))
u_noisy2 = u2 + σ2 .* randn(size(u))
tsteps = sol_v2.t
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (s)",
    ylabel="Concentration (mol L⁻¹ or kg⁻¹)",
    width=600, height=400)
#lines!(ax, sol_v2.t, [u[5] for u in sol_v2.u], label="C_c (v2)", color=:orange, linestyle=:dash)
lines!(ax, sol_v2.t, [u[1] for u in sol_v2.u], label="C1 - ode", color=:blue, linestyle=:dash)
lines!(ax, sol_v2.t, [u[2] for u in sol_v2.u], label="C2 - ode", color=:lightblue, linestyle=:dash)
lines!(ax, sol_v2.t, [u[3] for u in sol_v2.u], label="S1 - ode", color=:red, linestyle=:dash)
lines!(ax, sol_v2.t, [u[4] for u in sol_v2.u], label="S2 - ode", color=:darkred, linestyle=:dash)
scatter!(ax, sol_v2.t, u_noisy1, label="C1 - noisy", color=:blue)
scatter!(ax, sol_v2.t, u_noisy2, label="C2 - noisy", color=:lightblue)
Legend(fig[1,2],ax, position=:rc)
resize_to_layout!(fig)
fig

function residuals(x)
    problem = ODEProblem(rhs, u0, tspan, x)
    sol = solve(problem, Rosenbrock23(); saveat = tsteps,
     reltol = 1e-9, abstol = 1e-9, maxiters = 10000)
    u = sol.u
    u1 = [ux[1] for ux in u]
    u2 = [ux[2] for ux in u]
    [(u1 .- u_noisy1); (u2 .- u_noisy2)]
end

function jac(x)
    return ForwardDiff.jacobian(residuals, x)
end

f_true = residuals(p)
cost_true = 0.5 * dot(p, p)  # True cost value

Random.seed!(1234)  # Set seed for reproducibility
# sample from an uniform distribution within the bounds
x0 = randn(size(p))*0.5.*p.+ p  # Initial guess
cost_x0 = 0.5 * dot(residuals(x0), residuals(x0))

# Running the qr_nlss_bounded_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.qr_nlss_bounded_trust_region(
    residuals, jac, x0, repeat([-Inf], inner=length(x0)), repeat([Inf], inner=length(x0));
    max_iter = 100, gtol = 1e-8
)
lb = repeat([-Inf], inner=length(x0))  # Lower bounds
ub = repeat([Inf], inner=length(x0))  # Upper bounds
converged = norm(g_opt, Inf) < 1e-4
bounds_satisfied = all(lb .<= a_opt .<= ub)
parameter_error = norm(a_opt - p)
cost = 0.5 * dot(f_opt, f_opt)  # Cost function value

results = []  # Store all results for comparison

result_qr = (
    a_true = p,
    a_opt = a_opt,
    converged = converged,
    bounds_satisfied = bounds_satisfied,
    parameter_error = parameter_error,
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = cost,
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "QR-based TR"
)
push!(results, result_qr)

# Running the qr_nlss_bounded_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.qr_nlss_bounded_trust_region_v2(
    residuals, jac, x0, repeat([-Inf], inner=length(x0)), repeat([Inf], inner=length(x0));
    max_iter = 100, gtol = 1e-8
)

converged = norm(g_opt, Inf) < 1e-4
bounds_satisfied = all(lb .<= a_opt .<= ub)
parameter_error = norm(a_opt - p)
cost = 0.5 * dot(f_opt, f_opt)  # Cost function value


result_qr_scaled = (
    a_true = p,
    a_opt = a_opt,
    converged = converged,
    bounds_satisfied = bounds_satisfied,
    parameter_error = parameter_error,
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = cost,
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "QR-based TR scaled"
)
push!(results, result_qr_scaled)
res_nl(u, p) = residuals(u)
nlprob = NonlinearLeastSquaresProblem(res_nl, x0, p)

nl_solvers = [
    (TrustRegion(), "NonlinearSolve TR"),
    (LevenbergMarquardt(), "NonlinearSolve LM"),
    (FastShortcutNLLSPolyalg(), "NonlinearSolve Polyalg")
]

for (solver, solver_name) in nl_solvers
    try
        nlsol = solve(nlprob, solver;maxiters = 100, show_trace = Val(true),
            trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(25))
        a_opt = nlsol.u
        f_opt = residuals(a_opt)
        g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
        iter = nlsol.stats.nsteps
        converged = norm(g_opt, Inf) < 1e-4
        bounds_satisfied = all(lb .<= a_opt .<= ub)
        parameter_error = norm(a_opt - p)
        cost = 0.5 * dot(f_opt, f_opt)  # Cost function value
        result_nl = (
        a_true = p,
        a_opt = a_opt,
        converged = converged,
        bounds_satisfied = bounds_satisfied,
        parameter_error = parameter_error,
        iterations = iter,
        initial_cost = 0.5 * dot(residuals(p), residuals(p)),
        cost = cost,
        cost_true = 0.5 * dot(residuals(p), residuals(p)),
        solver = solver_name,
        )
        push!(results, result_nl)
    catch e
        result_nl = (
            a_true = p,
            a_opt = [NaN, NaN, NaN, NaN],
            converged = false,
            bounds_satisfied = false,
            parameter_error = NaN,
            iterations = NaN,
            initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
            cost = NaN,
            cost_true = 0.5 * dot(residuals(p), residuals(p)),
            solver = solver_name,
        )
        push!(results, result_nl)
    end
    # Store the result
end
cost_f(x) = 0.5 * dot(residuals(x), residuals(x))
res_prima = PRIMA.newuoa(
    cost_f, x0;
    maxfun = 500,
)

a_opt = res_prima[1]
f_opt = residuals(a_opt)
g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
iter = res_prima[2].nf
println("Prima info: $(PRIMA.reason(res_prima[2]))")
result_prima = (
    a_true = p,
    a_opt = a_opt,
    converged = res_prima[2] == PRIMA.SMALL_TR_RADIUS || res_prima[2] == PRIMA.FTARGET_ACHIEVED,
    bounds_satisfied = all(lb .<= a_opt .<= ub),
    parameter_error = norm(a_opt - p),
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = 0.5 * dot(f_opt, f_opt),  # Cost function value
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "PRIMA newuoa"
)
push!(results, result_prima)

# Check the result with an OPTIMIZARION solver

cost_o(u, p) = 0.5 * dot(residuals(u), residuals(u))
f_o = OptimizationFunction(cost_o, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(f_o, x0, p)
optsol = Optimization.solve(optprob, Optimization.LBFGS(); maxiters = 100)
a_opt = optsol.u
f_opt = residuals(a_opt)
g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
iter = optsol.stats.iterations
result_opt = (
    a_true = p,
    a_opt = a_opt,
    converged = norm(g_opt, Inf) < 1e-4,
    bounds_satisfied = all(lb .<= a_opt .<= ub),
    parameter_error = norm(a_opt - p),
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = 0.5 * dot(f_opt, f_opt),  # Cost function value
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "Optimization LBFGS"
)
push!(results, result_opt)
println("\n" * "="^40)
println("PERFORMANCE SUMMARY")
println("="^40)
# Header
println(rpad("Solver", 20) * rpad("Parameter Error", 16) * 
    rpad("Iterations", 12) * rpad("Cost", 12) * rpad("Converged", 10))
println("-"^40)

# Results for this ODE problem
for result in results
    println(rpad(result.solver, 20) * 
       rpad(round(result.parameter_error, digits=4), 16) *
       rpad(result.iterations, 12) *
       rpad(result.cost, 12) *
       rpad(result.converged ? "✓" : "✗", 10))
end

println()

# Best performer by accuracy
converged_results = filter(r -> r.converged, results)
if !isempty(converged_results)
    best_accuracy = minimum(r -> r.parameter_error, converged_results)
    best_solver = filter(r -> r.parameter_error == best_accuracy, converged_results)[1]
    println("Best Accuracy: $(best_solver.solver) (Error: $(round(best_accuracy, digits=4)))")
end

# Best performer by speed
if !isempty(converged_results)
    fewest_iters = minimum(r -> r.iterations, converged_results)
    fastest_solver = filter(r -> r.iterations == fewest_iters, converged_results)[1]
    println("Fastest: $(fastest_solver.solver) ($(fewest_iters) iterations)")
end

println("="^80)

# Bounded performance

smax = 0.0034 # Maximum sorbed concentration
K1 = 1e-3 # Equilibrium constant
K2 = 2e-4 # Equilibrium constant
λ = 0.1e-2 # Rate constant for sorption
p = [smax, K1, K2, λ]
lb = [smax* 0.3, K1*0.3, K2*0.3, λ*0.3]  # Lower bounds
ub = [smax* 1.8, K1*1.8, K2*1.8, λ*1.8]  # Upper bounds

# Running the bounded trust region solver

# Running the qr_nlss_bounded_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.qr_nlss_bounded_trust_region(
    residuals, jac, x0, lb, ub;
    max_iter = 100, gtol = 1e-8
)
converged = norm(g_opt, Inf) < 1e-4
bounds_satisfied = all(lb .<= a_opt .<= ub)
parameter_error = norm(a_opt - p)
cost = 0.5 * dot(f_opt, f_opt)  # Cost function value

results = []  # Store all results for comparison

result_qr = (
    a_true = p,
    a_opt = a_opt,
    converged = converged,
    bounds_satisfied = bounds_satisfied,
    parameter_error = parameter_error,
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = cost,
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "QR-based TR"
)
push!(results, result_qr)

# Running the qr_nlss_bounded_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.qr_nlss_bounded_trust_region_v2(
    residuals, jac, x0, lb, ub;
    max_iter = 100, gtol = 1e-8
)

converged = norm(g_opt, Inf) < 1e-4
bounds_satisfied = all(lb .<= a_opt .<= ub)
parameter_error = norm(a_opt - p)
cost = 0.5 * dot(f_opt, f_opt)  # Cost function value


result_qr_scaled = (
    a_true = p,
    a_opt = a_opt,
    converged = converged,
    bounds_satisfied = bounds_satisfied,
    parameter_error = parameter_error,
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = cost,
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "QR-based TR scaled"
)
push!(results, result_qr_scaled)
res_nl(u, p) = residuals(u)
nlprob = NonlinearLeastSquaresProblem(res_nl, x0, p)

nl_solvers = [
    (TrustRegion(), "NonlinearSolve TR"),
    (LevenbergMarquardt(), "NonlinearSolve LM"),
    (FastShortcutNLLSPolyalg(), "NonlinearSolve Polyalg")
]

for (solver, solver_name) in nl_solvers
    try
        nlsol = solve(nlprob, solver;maxiters = 100, show_trace = Val(true),
            trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(25))
        a_opt = nlsol.u
        f_opt = residuals(a_opt)
        g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
        iter = nlsol.stats.nsteps
        converged = norm(g_opt, Inf) < 1e-4
        bounds_satisfied = all(lb .<= a_opt .<= ub)
        parameter_error = norm(a_opt - p)
        cost = 0.5 * dot(f_opt, f_opt)  # Cost function value
        result_nl = (
        a_true = p,
        a_opt = a_opt,
        converged = converged,
        bounds_satisfied = bounds_satisfied,
        parameter_error = parameter_error,
        iterations = iter,
        initial_cost = 0.5 * dot(residuals(p), residuals(p)),
        cost = cost,
        cost_true = 0.5 * dot(residuals(p), residuals(p)),
        solver = solver_name,
        )
        push!(results, result_nl)
    catch e
        result_nl = (
            a_true = p,
            a_opt = [NaN, NaN, NaN, NaN],
            converged = false,
            bounds_satisfied = false,
            parameter_error = NaN,
            iterations = NaN,
            initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
            cost = NaN,
            cost_true = 0.5 * dot(residuals(p), residuals(p)),
            solver = solver_name,
        )
        push!(results, result_nl)
    end
    # Store the result
end
cost_f(x) = 0.5 * dot(residuals(x), residuals(x))
res_prima = PRIMA.bobyqa(
    cost_f, x0;
    xl = lb, xu = ub,
    maxfun = 500,
)

a_opt = res_prima[1]
f_opt = residuals(a_opt)
g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
iter = res_prima[2].nf
println("Prima info: $(PRIMA.reason(res_prima[2]))")
result_prima = (
    a_true = p,
    a_opt = a_opt,
    converged = res_prima[2] == PRIMA.SMALL_TR_RADIUS || res_prima[2] == PRIMA.FTARGET_ACHIEVED,
    bounds_satisfied = all(lb .<= a_opt .<= ub),
    parameter_error = norm(a_opt - p),
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = 0.5 * dot(f_opt, f_opt),  # Cost function value
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "PRIMA bobyqa"
)
push!(results, result_prima)

# Check the result with an OPTIMIZARION solver

cost_o(u, p) = 0.5 * dot(residuals(u), residuals(u))
f_o = OptimizationFunction(cost_o, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(f_o, x0, p; lb = lb, ub = ub)
optsol = Optimization.solve(optprob, Optimization.LBFGS(); maxiters = 100)
a_opt = optsol.u
f_opt = residuals(a_opt)
g_opt = jac(a_opt)' * f_opt  # Gradient of 0.
iter = optsol.stats.iterations
result_opt = (
    a_true = p,
    a_opt = a_opt,
    converged = norm(g_opt, Inf) < 1e-4,
    bounds_satisfied = all(lb .<= a_opt .<= ub),
    parameter_error = norm(a_opt - p),
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = 0.5 * dot(f_opt, f_opt),  # Cost function value
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "Optimization LBFGS"
)
push!(results, result_opt)
println("\n" * "="^40)
println("PERFORMANCE SUMMARY")
println("="^40)
# Header
println(rpad("Solver", 20) * rpad("Parameter Error", 16) * 
    rpad("Iterations", 12) * rpad("Cost", 12) * rpad("Converged", 10))
println("-"^40)

# Results for this ODE problem
for result in results
    println(rpad(result.solver, 20) * 
       rpad(round(result.parameter_error, digits=4), 16) *
       rpad(result.iterations, 12) *
       rpad(result.cost, 12) *
       rpad(result.converged ? "✓" : "✗", 10))
end

println()

# Best performer by accuracy
converged_results = filter(r -> r.converged, results)
if !isempty(converged_results)
    best_accuracy = minimum(r -> r.parameter_error, converged_results)
    best_solver = filter(r -> r.parameter_error == best_accuracy, converged_results)[1]
    println("Best Accuracy: $(best_solver.solver) (Error: $(round(best_accuracy, digits=4)))")
end

# Best performer by speed
if !isempty(converged_results)
    fewest_iters = minimum(r -> r.iterations, converged_results)
    fastest_solver = filter(r -> r.iterations == fewest_iters, converged_results)[1]
    println("Fastest: $(fastest_solver.solver) ($(fewest_iters) iterations)")
end

println("="^80)