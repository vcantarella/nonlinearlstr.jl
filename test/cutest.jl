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
fur(x) = obj(nlp, x)
gur(x) = grad(nlp, x)
H(x) = NLPModels.hess(nlp, x)

opt = nonlinearlstr.bounded_trust_region(fur, gur, H,x0, lb, ub,)
x = opt[1]
fur(x0)
fur(x)


stats = tron(nlp)
stats.objective
stats.iter
fur(stats.solution)
propertynames(stats)

scipy = pyimport("scipy.optimize")

# make a sequence of bounds
bounds = []
for i in 1:length(lb)
    push!(bounds, (lb[i], ub[i]))
end
bounds
sol_py = scipy.minimize(fur, x0, method="SLSQP", jac=gur, hess=H, bounds=bounds)
x_py = pyconvert(Vector, sol_py.x)
fur(x_py)
res = PRIMA.bobyqa(fur, x0, xl=lb, xu=ub, rhobeg = 0.001)
x_p = res[1]
fur(x_p)
finalize(nlp)