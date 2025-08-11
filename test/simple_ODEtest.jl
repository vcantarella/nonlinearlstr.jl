using Test
using LinearAlgebra
using Statistics
using Random
using ForwardDiff
using Pkg
using Revise
using NonlinearSolve
using OrdinaryDiffEq
using PRIMA
using Optimization
using CairoMakie

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

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
sol = solve(prob, Tsit5(); saveat = tsteps, abstol = 1e-9, reltol = 1e-9)

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

function residuals(x)
    problem = ODEProblem(lotka_volterra!, u0, tspan, x)
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

# Define the bounds
lb = [0.0, 0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
# sample from an uniform distribution within the bounds
x0 = [1, 1.5, 2, 0.5]
cost_x0 = 0.5 * dot(residuals(x0), residuals(x0))

# Running the lm_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region(
    residuals, jac, x0;
    max_iter = 100, gtol = 1e-8
)

converged = norm(g_opt, 2) < 1e-8
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

# Running the lm_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.lm_trust_region_scaled(
    residuals, jac, x0;
    max_iter = 400, gtol = 1e-8
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
    (NonlinearSolve.LevenbergMarquardt(), "NonlinearSolve LM"),
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
       rpad(round(result.cost, digits=6), 12) *
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

# Save table to CSV
df = DataFrame(results)
CSV.write("performance_summary_unbounded.csv", df)

# Plot results per solver

plot_title = "Solver model for a given ODE problem"
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time", ylabel="Population", title=plot_title)
line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash]
for (i, result) in enumerate(results)
    if result.a_opt == [NaN, NaN, NaN, NaN]
        continue  # Skip if no valid result
    end
    prob = ODEProblem(lotka_volterra!, u0, tspan, result.a_opt, 
        reltol = 1e-9, abstol = 1e-9)
    sol = solve(prob, Rosenbrock23())
    lines!(ax, sol.t, [sol.u[i][1] for i in eachindex(sol.u)], label="Predator ($(result.solver))",
        color=:blue, linestyle=line_styles[i], alpha=0.7)
    lines!(ax, sol.t, [sol.u[i][2] for i in eachindex(sol.u)], label="Prey ($(result.solver))",
        color=:red, linestyle=line_styles[i], alpha=0.7)
end
scatter!(ax, tsteps, u_noisy1, color=:blue, markersize=12, label="Noisy Predator Data")
scatter!(ax, tsteps, u_noisy2, color=:red, markersize=12, label="Noisy Prey Data")
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5();)
lines!(ax, sol.t, [sol.u[i][1] for i in eachindex(sol.u)], color=:blue, linestyle=:solid, label="True Predator")
lines!(ax, sol.t, [sol.u[i][2] for i in eachindex(sol.u)], color=:red, linestyle=:solid, label="True Prey")
Legend(fig[1, 2], ax)
fig
save("solver_model_ODE_unbounded.png", fig)

# Now lets check the problem with BOUNDS!!



# Running the qr_nlss_bounded_trust_region solver
a_opt, f_opt, g_opt, iter = nonlinearlstr.qr_nlss_bounded_trust_region(
    residuals, jac, x0, lb, ub;
    max_iter = 100, gtol = 1e-8
)

converged = norm(g_opt, Inf) < 1e-4
bounds_satisfied = all(lb .<= a_opt .<= ub)
parameter_error = norm(a_opt - p)
cost = 0.5 * dot(f_opt, f_opt)  # Cost function value

bounds_results = []  # Store all results for comparison

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
push!(bounds_results, result_qr)


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
push!(bounds_results, result_qr_scaled)

res_prima = PRIMA.bobyqa(
    cost_f, x0, xl=lb, xu=ub;
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
    converged = (res_prima[2].status == PRIMA.SMALL_TR_RADIUS) || (res_prima[2].status == PRIMA.FTARGET_ACHIEVED),
    bounds_satisfied = all(lb .<= a_opt .<= ub),
    parameter_error = norm(a_opt - p),
    iterations = iter,
    initial_cost = 0.5 * dot(residuals(x0), residuals(x0)),
    cost = 0.5 * dot(f_opt, f_opt),  # Cost function value
    cost_true = 0.5 * dot(residuals(p), residuals(p)),
    solver = "PRIMA bobyqa"
)
push!(bounds_results, result_prima)

optprob = OptimizationProblem(f_o, x0, p; lb=lb, ub=ub)
optsol = Optimization.solve(optprob,
    Optimization.LBFGS(); maxiters = 100)
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

push!(bounds_results, result_opt)

println("\n" * "="^40)
println("PERFORMANCE SUMMARY")
println("="^40)
# Header
println(rpad("Solver", 20) * rpad("Parameter Error", 16) * 
    rpad("Iterations", 12) * rpad("Cost", 12) * rpad("Converged", 10))
println("-"^40)

# Results for this ODE problem
for result in bounds_results
    println(rpad(result.solver, 20) * 
       rpad(round(result.parameter_error, digits=4), 16) *
       rpad(result.iterations, 12) *
       rpad(round(result.cost, digits=6), 12) *
       rpad(result.converged ? "✓" : "✗", 10))
end
println()



# Save table to CSV
df = DataFrame(bounds_results)
CSV.write("performance_summary_bounded.csv", df)

# Plot results per solver
plot_title = "Solver model for a given ODE problem - Bounds"
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time", ylabel="Population", title=plot_title)
line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
for (i, result) in enumerate(bounds_results)
    prob = ODEProblem(lotka_volterra!, u0, tspan, result.a_opt)
    sol = solve(prob, Tsit5())
    lines!(ax, sol.t, [sol.u[i][1] for i in eachindex(sol.u)], label="Predator ($(result.solver))",
        color=:blue, linestyle=line_styles[i], alpha=0.7)
    lines!(ax, sol.t, [sol.u[i][2] for i in eachindex(sol.u)], label="Prey ($(result.solver))",
        color=:red, linestyle=line_styles[i], alpha=0.7)
end
scatter!(ax, tsteps, u_noisy1, color=:blue, markersize=12, label="Noisy Predator Data")
scatter!(ax, tsteps, u_noisy2, color=:red, markersize=12, label="Noisy Prey Data")
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5();)
lines!(ax, sol.t, [sol.u[i][1] for i in eachindex(sol.u)], color=:blue, linestyle=:solid, label="True Predator")
lines!(ax, sol.t, [sol.u[i][2] for i in eachindex(sol.u)], color=:red, linestyle=:solid, label="True Prey")
Legend(fig[1, 2], ax)
fig
save("solver_model_ODE_bounded.png", fig)