using DifferentialEquations
using SciMLSensitivity
using ForwardDiff

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
u = sol.u
u1 = [ux[1] for ux in u]
σ1 = 0.1 * u1
u2 = [ux[2] for ux in u]
σ2 = 0.1 * u2
u_noisy1 = u1 + σ1 .* randn(size(u))
u_noisy2 = u2 + σ2 .* randn(size(u))

# Plot the solution
using Plots
plot(sol; linewidth = 3)
scatter!(tsteps, u_noisy1, yerror = σ1, label = "x noisy", color = :red)
scatter!(tsteps, u_noisy2, yerror = σ2, label = "y noisy", color = :blue)

# Define the cost function
function cost(p)
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat = tsteps)
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
hess(p) = ForwardDiff.jacobian(grad_cost, p)

# Define the bounds
lb = [0.0, 0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
# sample from an uniform distribution within the bounds
x0 = [rand() * (ub[i] - lb[i]) + lb[i] for i in 1:4]

# Run the optimization
include("../src/nonlinear_lstr.jl")
opt = nonlinearlstr.bounded_trust_region(cost, grad_cost, hess,x0, lb, ub,
        initial_radius = 1e-5)
x = opt[1]
cost(x)

function resi(x)
    problem = ODEProblem(lotka_volterra!, u0, tspan, x)
    sol = solve(problem, Tsit5(); saveat = tsteps)
    u = sol.u
    u1 = [ux[1] for ux in u]
    u2 = [ux[2] for ux in u]
    [u_noisy1 .- u1; u_noisy2 .- u2]
    
end

function jac(x)
    return ForwardDiff.jacobian(resi, x)
end

nls = nonlinearlstr.nlss_bounded_trust_region(resi, jac, x0, lb, ub)


cost(x0)
using PRIMA
res = PRIMA.bobyqa(cost, x0, xl=lb, xu=ub)
x_p = res[1]
cost(x_p)
# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, x)
sol = solve(prob, Tsit5())
plot(sol; linewidth = 3, color = [:red :blue])
scatter!(tsteps, u_noisy1, yerror = σ1, label = "x noisy", color = :red)
scatter!(tsteps, u_noisy2, yerror = σ2, label = "y noisy", color = :blue)