using CUTEst
using NLPModels
using JSOSolvers
include("../src/nonlinearlstr.jl")
using PRIMA
using PythonCall
using NonlinearSolve

# Load the CUTEst problem
available_problems = list_sif_problems()
filtered_problems = CUTEst.select(objtype="sum_of_squares", contype="bounds")
nlp = CUTEstModel(filtered_problems[3])
println("x0 = $( nlp.meta.x0 )")


println("fx = $( obj(nlp, nlp.meta.x0) )")
println("gx = $( grad(nlp, nlp.meta.x0) )")
println("Hx = $( NLPModels.hess(nlp, nlp.meta.x0) )")

# Solve the problem with our method and compare with other solvers
x0 = nlp.meta.x0
ilb = nlp.meta.jlow
iub = nlp.meta.jupp
irng = nlp.meta.iupp
lb = nlp.meta.lvar
ub = nlp.meta.uvar
x0[x0 .< lb .+ 1e-12] .= lb[x0 .< lb .+ 1e-12] .+ 0.5.*min.(1, ub[x0 .< lb .+ 1e-12] .- lb[x0 .< lb .+ 1e-12])
x0[x0 .> ub .- 1e-12] .= ub[x0 .> ub .- 1e-12] .- 0.5.*min.(1, ub[x0 .> ub .- 1e-12] .- lb[x0 .> ub .- 1e-12])
fur(x) = obj(nlp, x)
gur(x) = grad(nlp, x)
H(x) = NLPModels.hess(nlp, x)

opt = nonlinearlstr.bounded_trust_region(fur, gur, H,x0, lb, ub,)
x = opt[1]
fur(x0)
fur(x)


stats = tron(nlp)
stats

scipy = pyimport("scipy.optimize")

# make a sequence of bounds
bounds = []
for i in 1:length(lb)
    push!(bounds, (lb[i], ub[i]))
end
bounds
sol_py = scipy.minimize(fur, x0, method="trust-constr", jac=gur, hess=H, bounds=bounds)
x_py = pyconvert(Vector, sol_py.x)
f(x_py)
res = PRIMA.bobyqa(fur, x0, xl=lb, xu=ub, rhobeg = 0.001)
x_p = res[1]
fur(x_p)
finalize(nlp)