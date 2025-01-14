using LinearAlgebra
"""
If the model Φ(x) is linear in the parameters, then the least squares problem is linear in the parameters.
The most robust solution is to compute the SVD factorization of the Jacobian matrix J:
    J = UΣV'
where U and V are orthogonal matrices and Σ is a diagonal matrix with the singular values of J.
then the solution to the least squares problem is given by:
    x = VΣ⁻¹U'y
where y is the vector of observations.
When J is rank deficient, the least squares problem is ill-posed and the solution is not unique.
then any vector x that satisfies the normal equations J'Jx = J'y is a solution to the least squares problem.
"""

"""
Now lets implement the linear least squares problem
"""


function linear_least_squares(J::Matrix, y::Vector, ϵ::Real=1e-16)
    U, Σ, Vt = svd(J)
    V = transpose(Vt)
    x = zeros(eltype(J), size(J, 2))
    for j in eachindex(Σ)
        if Σ[j] > ϵ
            x += ((U[:, j]'*y) / Σ[j]) .* Vt[:, j]
        end
    end
    #x = V * (Σ \ (U' * y))
    return x
end

function lls(J::Matrix, y::Vector)
    U, Σ, Vt = svd(J)
    V = transpose(Vt)
    Σ = diagm(Σ)
    x = V' * (Σ\(U' * y))
    return x
end
