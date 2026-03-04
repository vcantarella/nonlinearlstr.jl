module nonlinearlstr
using LinearAlgebra
include("types.jl")
include("scaling.jl")
include("caches.jl")
include("subproblems.jl")
include("evd_subproblem.jl")
include("bounded_subproblems.jl")
include("bounded_step.jl")
include("algorithms.jl")
include("bounded_algorithms.jl")

# new versions
include("caches_v2.jl")
include("subproblem_mix.jl")
include("algorithms_v2.jl")

export lm_trust_region
export lm_trust_region_reflective
export lm_trust_region_v2!
end
