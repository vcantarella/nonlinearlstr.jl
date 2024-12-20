using DifferentialEquations
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
    sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-9, abstol = 1e-9)
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
hess(p) = grad_cost(p)' * grad_cost(p)

# Define the bounds
lb = [0.0, 0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
# sample from an uniform distribution within the bounds
x0 = [rand() * (ub[i] - lb[i]) + lb[i] for i in 1:4]
cost(x0)
# Run the optimization
include("../src/nonlinearlstr.jl")
opt = nonlinearlstr.bounded_trust_region(cost, grad_cost, hess,x0, lb, ub,
        initial_radius = 1e-5)
x = opt[1]
cost(x)

function resi(x)
    problem = ODEProblem(lotka_volterra!, u0, tspan, x)
    sol = solve(problem, Rosenbrock23(); saveat = tsteps)
    u = sol.u
    u1 = [ux[1] for ux in u]
    u2 = [ux[2] for ux in u]
    [(u_noisy1 .- u1)./σ1.^2; (u_noisy2 .- u2)./σ2.^2]
    
end

function jac(x)
    return ForwardDiff.jacobian(resi, x)
end

nls = nonlinearlstr.nlss_bounded_trust_region(resi, jac, x0, lb, ub,
        initial_radius = 1)

x_nls = nls[1]
cost(x_nls)
using PRIMA
res = PRIMA.bobyqa(cost, x0, xl=lb, xu=ub)
x_p = res[1]
cost(x_p)

using NonlinearSolve
res_2(x, p) = resi(x)
nlls_prob = NonlinearProblem(res_2, x0)
nlls_sol = solve(nlls_prob, TrustRegion(initial_trust_radius = 1e-6);
     maxiters = 1000, show_trace = Val(true),
trace_level = NonlinearSolve.TraceWithJacobianConditionNumber(25))
p_nl = nlls_sol.u
cost(p_nl)
# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, x_p)
sol = solve(prob, Tsit5())
plot(sol; linewidth = 3, color = [:red :blue])
scatter!(tsteps, u_noisy1, yerror = σ1, label = "x noisy", color = :red)
scatter!(tsteps, u_noisy2, yerror = σ2, label = "y noisy", color = :blue)

using PythonCall

scipy = pyimport("scipy.optimize")

pyls = scipy.least_squares(resi, x0, jac=jac, bounds=(lb, ub), xtol = 1e-16)
x_py = pyconvert(Vector, pyls.x)
cost(x_py)
cost(x0)
# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, x_py)
sol = solve(prob, Tsit5())
plot(sol; linewidth = 3, color = [:red :blue])
scatter!(tsteps, u_noisy1, yerror = σ1, label = "x noisy", color = :red)
scatter!(tsteps, u_noisy2, yerror = σ2, label = "y noisy", color = :blue)


# Setting up non linear least squares experiment

# define a random seed
using Random
Random.seed!(1234)

# define the number of data points
n = 100
distance = zeros((n, 5))
costs = zeros((n,5))
successes = zeros(Int,(n,5))
# for loop generating the realizations
for i in 1:n
    # generate the noisy data
    u_noisy1 = u1 + σ1 .* randn(size(u))
    u_noisy2 = u2 + σ2 .* randn(size(u))
    # define the cost function
    function cost(p)
        prob = ODEProblem(lotka_volterra!, u0, tspan, p)
        sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-9, abstol = 1e-9)
        u = sol.u
        u1 = [ux[1] for ux in u]
        u2 = [ux[2] for ux in u]
        cost1 = sum((u_noisy1 .- u1).^2 ./ σ1.^2)
        cost2 = sum((u_noisy2 .- u2).^2 ./ σ2.^2)
        return cost1 + cost2
    end

    # define the gradient of the cost function
    function grad_cost(p)
        return ForwardDiff.gradient(cost, p)
    end

    # define the hessian of the cost function
    hess(p) = ForwardDiff.jacobian(grad_cost, p)

    # define the bounds
    lb = [0.0, 0.0, 0.0, 0.0]
    ub = [10.0, 10.0, 10.0, 10.0]
    # sample from an uniform distribution within the bounds
    x0 = [rand() * (ub[i] - lb[i]) + lb[i] for i in 1:4]

    try
        # run the optimization
        opt = nonlinearlstr.bounded_trust_region(cost, grad_cost, hess,x0, lb, ub,
                initial_radius = 1e-5)
        x = opt[1]
        costs[i, 1] = cost(x)
        distance[i,1] = norm(p-x)
        successes[i,1] = 1
    catch
        distance[i,1] = NaN
        costs[i,1] = NaN
        successes[i,1] = 0
    end

    try
        # run the optimization
        nls = nonlinearlstr.nlss_bounded_trust_region(resi, jac, x0, lb, ub,
                initial_radius = 1e-9)

        x_nls = nls[1]
        costs[i,2] = cost(x_nls)
        distance[i,2] = norm(p-x_nls)
        successes[i,2] = 1
    catch
        distance[i,2] = NaN
        costs[i,2] = NaN
        successes[i,2] = 0
    end

    try
        # run the optimization
        res = PRIMA.bobyqa(cost, x0, xl=lb, xu=ub)
        x_p = res[1]
        costs[i,3] = cost(x_p)
        distance[i,3] = norm(p-x_p)
        successes[i,3] = 1
    catch
        distance[i,3] = NaN
        costs[i,3] = NaN
        successes[i,3] = 0
    end

    try
    res_2(x, p) = resi(x)
    nlls_prob = NonlinearProblem(res_2, x0)
    nlls_sol = solve(nlls_prob, TrustRegion(initial_trust_radius = 1e-9);
        maxiters = 1000, show_trace = Val(true),
    trace_level = TraceWithJacobianConditionNumber(25))
    p_nl = nlls_sol.u
    costs[i,4] = cost(p_nl)
    distance[i,4] = norm(p-p_nl)
    successes[i,4] = 1
    catch
        distance[i,4] = NaN
        costs[i,4] = NaN
        successes[i,4] = 0
    end

    try
    pyls = scipy.least_squares(resi, x0, jac=jac, bounds=(lb, ub), xtol = 1e-16)
    x_py = pyconvert(Vector, pyls.x)
    costs[i,5] = cost(x_py)
    distance[i,5] = norm(p-x_py)
    successes[i,5] = 1
    catch
        distance[i,5] = NaN
        costs[i,5] = NaN
        successes[i,5] = 0
    end
end

# plot boxplots of the results
using StatsPlots
# Remove NaN values for plotting
# instead of plotting the full matrix. plot the vectors filtered individually
clean_costs = [costs[findall(.!isnan.(costs[:,i])),i] for i in 1:5]
clean_distance = [distance[findall(.!isnan.(distance[:,i])),i] for i in 1:5]
# Plot boxplots of the results
xlabels = ["nonlinearlstr", "nlss", "PRIMA", "NonlinearSolve", "scipy"]
plt = Plots.plot()
for (i,cost) in enumerate(clean_costs)
    boxplot!(plt, [xlabels[i]], cost, label = xlabels[i], legend = :topleft, outliers = false)
end
plt
boxplot!(plt, ["nonlinearlstr", "nlss", "PRIMA", "NonlinearSolve", "scipy"], clean_costs, label = "cost", legend = :topleft, outliers = false)

plt2 = Plots.plot()
for (i,dist) in enumerate(clean_distance)
    boxplot!(plt2, [xlabels[i]], dist, label = xlabels[i], legend = :topleft, outliers = false)
end
plt2

# Bar plot for the success rate
boxplot(["nonlinearlstr", "nlss", "PRIMA", "NonlinearSolve", "scipy"], costs', label = "cost", legend = :topleft)
boxplot(["nonlinearlstr", "nlss", "PRIMA", "NonlinearSolve", "scipy"], distance, label = "distance", legend = :topleft)

# bar plot for the success rate
bar(["nonlinearlstr", "nlss", "PRIMA", "NonlinearSolve", "scipy"], sum(successes, dims = 1)', label = "success", legend = :topleft)