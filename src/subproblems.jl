abstract type SubProblemStrategy end
struct QR <: SubProblemStrategy end
struct SVD <: SubProblemStrategy end

factorize(::QR, J) = qr(J)
factorize(::SVD, J) = svd(J)

function solve_subproblem(strategy::SubProblemStrategy, J::AbstractMatrix{T},
                          f::AbstractVector{T}, Dk::Diagonal{T},
                          radius::Real, cache) where {T<:Real}
    F = get!(cache, typeof(strategy)) do
        factorize(strategy, J)
    end
    δgn = F \ -f
    if norm(δgn) <= radius
        λ = zero(T)
        δ = δgn
    else
        λ, δ = find_λ_scaled(strategy, radius, J, f, Dk, 100)
    end
    return λ, δ
end



function find_λ_scaled(strategy, Δ, J, D, f, maxiters, θ=1e-4)
    l₀ = 0.0
    u₀ = norm(D*J'f)/Δ
    λ₀ = max(1e-3*u₀,√(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    b_aug = [-f; zeros(size(J, 2))]
    for i in 1:maxiters
        # F = factorize(strategy, [J;√λ*D])
        # p = F \ b_aug
        p = solve_augmented(strategy, J, D, b_aug, λ)
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = λ
        end
        dpdλ = solve_for_dp_dlambda_scaled(strategy, qrf, p, D)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D'*(D*dpdλ)))
        if !(uₖ < λ <= lₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ),√(lₖ*uₖ))
        end
    end
    println("Final step norm: ", norm(D*p))
    return λ, p
end

function solve_augmented(::QR, J::AbstractMatrix, D::Diagonal, b_aug::AbstractVector, λ::Real)
    J_aug = [J; sqrt(λ)*D]
    qrf = qr(J_aug)
    δ = qrf \ b_aug
    return δ
end

function solve_augmented(::SVD, J::AbstractMatrix, D::Diagonal, b_aug::AbstractVector, λ::Real)
    svdls = svd(J)
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    δ = zeros(size(J, 2))
    for i in 1:n
        δ += (σs[i]/(σs[i]^2 + λ*D[i,i]^2))*(U[:,i]'*b_aug)*V[:,i]
    end
    return δ
end

function solve_for_dp_dlambda_scaled(::QR, qrf, p::AbstractVector, D::AbstractMatrix)
    # Perform the QR factorization to get the factors explicitly
    R = qrf.R
    rhs = -(D'*(D*p))
    # Now solve (RᵀR) * (dp/dλ) = -p
    # This is done in two steps:
    # 1. Rᵀz = -p  =>  z = Rᵀ \ -p
    # 2. R(dp/dλ) = z  =>  dp/dλ = R \ z
    dp_dλ = UpperTriangular(R) \ (LowerTriangular(R') \ rhs)
    return dp_dλ
end

function solve_for_dp_dlambda_scaled(::SVD, svdls, p::AbstractVector, D::AbstractMatrix)
    # Perform the QR factorization to get the factors explicitly
    U = svdls.U
    V = svdls.V
    σs = svdls.S
    n = length(σs)
    dpdλ = sum([-σs[i]/(σs[i]^2 + λ)^2*U[:,i]'*b*V[:,i] for i in 1:n])
    return dpdλ
end