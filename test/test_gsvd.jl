using LinearAlgebra
using nonlinearlstr
using Test
J = randn(10, 4)
b = randn(10)
λ = 1.0

function calculate_clambda(F1::GeneralizedSVD, λ::Real)
    # Extract the diagonal values from C₁ and S₁
    c1_diag = diag(F1.D1)
    
    # F1.D2 has a special structure; its diagonal is in the top-right block.
    # We need to extract the corresponding diagonal elements.
    # The size of the diagonal is the number of columns in D1.
    k_l = size(F1.D1, 2)
    s1_diag = diag(F1.D2[1:k_l, (end-k_l+1):end])

    # Calculate the diagonal of Cλ using the formula from the paper [cite: 600]
    clambda_diag = c1_diag ./ sqrt.(c1_diag.^2 .+ λ^2 .* s1_diag.^2)
    
    # Return Cλ as a diagonal matrix
    return diagm(clambda_diag)
end


# 1. Calculate the unregularized solution, x₀ 
x0 = J \ b

# 2. Perform the GSVD for λ=1 to get the base components C₁ and S₁ [cite: 599]
F1 = svd(J, I(n))

# 3. Calculate Cλ for our specific λ
Clambda = calculate_clambda(F1, 1.0)

# 4. Determine U and H₀ from the relation A = UH₀ 
# U is the orthogonal matrix from the λ=1 GSVD.
U = F1.U
# We can find H₀ via the pseudoinverse: H₀ = U⁺A, which is U'A since U is orthogonal.
H0 = U' * J

# 5. Assemble the final formula for xλ 
# Use the backslash operator for inv(H₀) for better numerical stability.
# Note the element-wise squaring of the Clambda matrix.
x_lambda_formula = H0 \ ( (Clambda .^ 2) * (H0 * x0) )

println("\nSolution using the paper's formula (Eq. 8.2):")
display(x_lambda_formula)



# --- Inputs ---
A = randn(10, 5) # m₁=10, n=5
L = I(5)
b = randn(10)
λ = 0.1

# --- Corrected Implementation of the Paper's Formula ---

# 1. Calculate the unregularized solution, x₀
# x₀ will have length n (5)
x0 = J \ b

# 2. Perform the GSVD for λ=1
F1 = svd(J, I(n))

# 3. Get the effective rank 'r' from the number of columns in D1
# This might be less than n (e.g., 4, as in your error)
r = size(F1.D1, 2)

# 4. Calculate Cλ, but keep only the compact, square, r x r part
# The original calculate_clambda function from before is still needed
Clambda_full = calculate_clambda(F1, λ) # This is m₁ x r (e.g., 10x4)
Clambda_compact = Clambda_full[1:r, 1:r] # Truncate to r x r (e.g., 4x4)

# 5. Get the COMPACT U by truncating to its first 'r' columns
# U must be m₁ x r (e.g., 10x4)
U_compact = F1.U[:, 1:r]

# 6. Calculate H₀ using the compact U. H₀ will now be r x n (e.g., 4x5)
H0 = U_compact' * J

# 7. Assemble the final formula. All dimensions now align correctly.
# (H0 * x0) is an r-element vector (e.g., 4x1)
# (Clambda_compact .^ 2) is r x r (e.g., 4x4)
# The result of the multiplication is an r-element vector
y = (Clambda_compact .^ 2) * (H0 * x0)

# Since H₀ is not square, use the pseudoinverse (pinv) to solve H₀*x = y
# The final result will be an n-element vector (e.g., 5x1)
x_lambda_formula = H0 \ y

println("\nSolution using the corrected formula implementation:")
display(x_lambda_formula)

# For comparison, the direct method should give a similar result
x_lambda_direct = [J; λ*I(n)] \ [b; zeros(size(I(n),1))]
println("\nSolution using direct method:")
display(x_lambda_direct)
@test all(x_lambda_formula .≈ x_lambda_direct)