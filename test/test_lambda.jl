using Test
using LinearAlgebra
using Statistics
using Random
using ForwardDiff
using Pkg
using Revise
using NonlinearSolve
using nonlinearlstr


# --- Example Usage ---
# Let's create some dummy data
m, n = 10, 4
J = randn(m, n)
f_true = randn(n)
p_true = randn(n) # This is not used, just for creating f
f = -J * p_true + 0.1 * randn(m) # Add some noise
λ = 0.1

# 1. Solve for p
# Remember your code solves for (J'J+λI)p = J'b, but we need (J'J+λI)p = -J'f
# So we set b = -f
function get_p(J, f, λ)
    n = size(J, 2)
    J_aug = [J; sqrt(λ) * I(n)]
    b = [-f; zeros(n)]
    return J_aug \ b
end

p = get_p(J, f, λ)

dp_dλ = nonlinearlstr.solve_for_dp_dlambda(J, p, λ)
dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)
# 2. Now that we have p, solve for dp/dλ
dp_dλ = solve_for_dp_dlambda(J, p, λ)

println("Solved for p:\n", p)
println("\nSolved for dp/dλ:\n", dp_dλ)
@test dp_dλ ≈ dp_dλ_fd atol=1e-6