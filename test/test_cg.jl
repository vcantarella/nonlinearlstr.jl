using LinearAlgebra
include("../src/cg.jl")
include("../src/trsbox.jl")
using Test

A = [3 0 1; 0 4 2; 1 2 3]
b = [-3, 0, -1]

x0 = [0, 0, 0]
tol = 1e-9
max_iter = 100
x = cg(A, b, x0, tol, max_iter)

x_inv = A\-b

@test x ≈ x_inv atol=1e-9

Δ = 1.0
l = [-10.0, -10.0, -10.0]
u = [10.0, 10.0, 10.0]
tol = 1e-9
max_iter = 1000
fx = 0.
x_trs = trsbox(A, b, Δ, fx, l, u, tol, max_iter)

@test x_trs ≈ x_inv atol=1e-9