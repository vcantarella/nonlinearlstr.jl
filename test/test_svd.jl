using LinearAlgebra
using Revise
using nonlinearlstr
using Test
using ForwardDiff
using Pkg
J = randn(10, 4)
b = randn(10)
λ = 0.5
n = size(J, 2)
J_aug = [J; sqrt(λ) * I(n)]
b_aug = [b; zeros(n)]
x_qr = J_aug \ b_aug
svdls = svd(J)
x_svd = nonlinearlstr.svd_regularized_solve(svdls, b, λ)

@test x_qr ≈ x_svd atol=1e-7

dpdλ_svd = nonlinearlstr.solve_for_dpdλ(svdls, b, λ)


dpdλ_fd = ForwardDiff.derivative(λ -> nonlinearlstr.svd_regularized_solve(svdls, b, λ), λ)

@test dpdλ_svd ≈ dpdλ_fd atol=1e-7

# interesting: my previous function has more error
dpdλ_qr = nonlinearlstr.solve_for_dp_dlambda(J_aug, x_qr, λ)

radius = 0.5
x0 = J \ b
if norm(x0) > radius
    λ, p = nonlinearlstr.find_λ_svd(radius, svdls, J, b, 100)
    @test λ > 0
    @test norm(p) ≈ radius atol=1e-4
end