using Revise
using LinearAlgebra
using Test
include("../src/colemanli.jl")

# Test the zero matrix for infinite bounds:
n = 10
x = randn(n)
lb = fill(-Inf, n)
ub = fill(Inf, n)
g = randn(n)
D, Jv = affine_scale_matrix(x, lb, ub, g)
@test D == I(n)
@test Jv == zeros(n, n)