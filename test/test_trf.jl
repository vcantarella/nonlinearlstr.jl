using LinearAlgebra
using Test
include("../src/trf_utils.jl")

x = [0.5, 0.1, 0.2]
l = -[0.3, 0.3, 0.3]
u = [0.7, 0.7, 0.7]

s1 = [0.3, 0.1, 0.1]

step, hits = step_size_to_bound(x, s1, l, u)

@test hits == [1, 0, 0]
@test step ≈ 0.2/0.3

s2 = [0.1, -0.4, 0.2]
step2, hits2 = step_size_to_bound(x, s2, l, u)

@test hits2 == [0, -1, 0]
@test step2 ≈ 1.0

s3 = [0.3, -0.45, 0.2]
step3, hits3 = step_size_to_bound(x, s3, l, u)

@test hits3 == [1, 0, 0]
@test step3 ≈ min(0.2/0.3, 1.0)