using Test
using LinearAlgebra
using ForwardDiff

# Include the module
include("../src/nonlinearlstr.jl")
using .nonlinearlstr

# Test QR-based nonlinear least squares on a simple problem
println("Testing QR-based Nonlinear Least Squares Solver")
println("=" ^ 50)

# Simple nonlinear least squares problem: fit exponential decay
# Model: y = a * exp(-b * t) + c
# Data generation
t_data = 0.0:0.5:5.0
a_true, b_true, c_true = 2.0, 0.5, 0.1
y_true = a_true * exp.(-b_true * t_data) .+ c_true
noise = 0.05 * randn(length(t_data))
y_data = y_true + noise

# Define residual function
function residuals(params)
    a, b, c = params
    y_model = [a * exp(-b * t) + c for t in t_data]
    return y_data - y_model
end

# Define Jacobian function
function jacobian(params)
    return ForwardDiff.jacobian(residuals, params)
end

# Initial guess and bounds
x0 = [1.5, 0.3, 0.05]
lb = [0.1, 0.01, -1.0]
ub = [5.0, 2.0, 1.0]

println("True parameters: a=$a_true, b=$b_true, c=$c_true")
println("Initial guess: ", x0)
println()

# Test QR-based solver
println("Testing QR-based solver:")
result_qr = nonlinearlstr.qr_nlss_bounded_trust_region(
    residuals, jacobian, x0, lb, ub;
    max_iter=50, gtol=1e-8, ftol=1e-12
)

x_qr, f_qr, g_qr, iter_qr = result_qr
cost_qr = 0.5 * dot(f_qr, f_qr)

println()
println("QR-based solver results:")
println("Final parameters: ", x_qr)
println("Final cost: ", cost_qr)
println("Final gradient norm: ", norm(g_qr, Inf))
println("Iterations: ", iter_qr)
println("Parameter errors: ", abs.(x_qr - [a_true, b_true, c_true]))

# Compare with your existing solver if available
println()
println("Comparing with existing solver:")
try
    result_old = nonlinearlstr.nlss_bounded_trust_region(
        residuals, jacobian, x0, lb, ub;
        max_iter=50, gtol=1e-8
    )
    
    x_old, f_old, g_old, iter_old = result_old
    cost_old = 0.5 * dot(f_old, f_old)
    
    println("Old solver results:")
    println("Final parameters: ", x_old)
    println("Final cost: ", cost_old)
    println("Final gradient norm: ", norm(g_old, Inf))
    println("Iterations: ", iter_old)
    println("Parameter errors: ", abs.(x_old - [a_true, b_true, c_true]))
    
    println()
    println("Improvement:")
    println("Cost ratio (QR/Old): ", cost_qr / cost_old)
    println("Iterations ratio (QR/Old): ", iter_qr / iter_old)
    
catch e
    println("Old solver not available or failed: $e")
end

# Test QR utilities directly
println()
println("Testing QR utilities:")
J_test = jacobian(x_qr)
qrls_test = nonlinearlstr.QRLeastSquares(J_test)

println("Jacobian condition number estimate: ", nonlinearlstr.condition_number(qrls_test))
println("Jacobian rank: ", qrls_test.rank, " / ", size(J_test, 2))

# Test Gauss-Newton step
gn_step = nonlinearlstr.solve_gauss_newton(qrls_test, f_qr)
println("Gauss-Newton step norm: ", norm(gn_step))

# Verify QR gradient computation
g_qr_direct = nonlinearlstr.compute_gradient(qrls_test, f_qr)
g_traditional = J_test' * f_qr
println("Gradient computation error: ", norm(g_qr_direct - g_traditional))

println()
println("Test completed successfully!")
