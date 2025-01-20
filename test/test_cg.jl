using LinearAlgebra
using BenchmarkTools
include("../src/cg.jl")
include("../src/trsbox.jl")
include("../src/tcg.jl")
include("../src/linear_least_squares.jl")
include("../src/projected_gradient.jl")
using Test

A = [3 0 1; 0 4 2; 1 2 3]
b = [-3, 0, -1]

x0 = [0, 0, 0]
tol = 1e-9
max_iter = 100
@btime x = cg($A, $b, $x0, $tol, $max_iter)

x_inv = A\-b

@test x ≈ x_inv atol=1e-9

Δ = 2.0
l = [-10.0, -10.0, -10.0]
u = [10.0, 10.0, 10.0]
tol = 1e-9
max_iter = 1000
fx = 0.
# @btime x_trs = trsbox(A, b, Δ, fx, l, u, tol, max_iter)
@btime x_tcg = tcg($A, $b, $Δ, $l, $u, $tol, $max_iter)
# @test x_trs ≈ x_inv atol=1e-9
x_tcg = tcg(A, b, Δ, l, u, tol, max_iter)
@test x_tcg ≈ x_inv atol=1e-9

@btime x_pcg = projected_cg($b, $A, $l, $u, $Δ, $max_iter, $tol)
@test x_pcg ≈ x_inv atol=1e-9

J = convert.(Float64, A)
y = convert.(Float64, -b)
x_lls = lls(J, y)
x_least = linear_least_squares(J, y)
@test x_lls ≈ x_inv atol=1e-9
@test x_least ≈ x_inv atol=1e-9

Q, R = qr(J)
lambda = 1e-6
R = cholesky(J'J + lambda.*diagm(diag(J'J)))
λ, Q = eigen(J'J)


