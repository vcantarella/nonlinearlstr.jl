
function find_λ_scaled_b(Δ, J, A, D, f, maxiters, θ = 1e-4)
    l₀ = 0.0
    u₀ = norm(D*J'f)/Δ
    n = size(J, 2)
    λ₀ = max(1e-3*u₀, √(l₀*u₀))
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    b_aug = [-f; zeros(size(J, 2))]
    for i = 1:maxiters
        qrf = qr([J; √Diagonal(λ*I(n)+A)*D])
        p = qrf \ b_aug
        if (1-θ)*Δ < norm(D*p) < (1+θ)*Δ
            break
        end
        ϕ = norm(D*p)-Δ
        if ϕ < 0
            uₖ = λ
        else
            lₖ = max(lₖ, λ)
        end
        dpdλ = solve_for_dp_dlambda_scaled(QRSolve(), qrf, p, D)
        λ = λ - (norm(D*p)-Δ)/Δ*((D*p)'*(D*p)/(p'*D*(D*dpdλ)))
        if !(lₖ < λ < uₖ)
            λ = max(lₖ+0.01*(uₖ-lₖ), √(lₖ*uₖ))
        end
    end
    println("Final step norm: ", norm(D*p))
    return λ, p
end
