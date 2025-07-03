using LinearAlgebra
"""
Levenberg-Marquardt optimization algorithm

"""

function levenberg_marquadt(x, f, J, λ, tol, max_iter)
    for i in 1:max_iter
        r = f(x)
        Jx = J(x)
        A = Jx'Jx + λ * I
        b = Jx'r
        Δx = A\b
        x += Δx
        if norm(Δx) < tol
            break
        end
    end
    return x
end

function levenberg_marquadtv2(x, f, J, λ, tol, max_iter)
    for i in 1:max_iter
        r = f(x)
        Jx = J(x)
        Dk = sum(Jk, dims = 1)
        Dk = diagm(Dk)
        A = Jx'Jx + λ * Dk
        b = -Jx'r
        Δx = A\b
        x += Δx
        if norm(Δx) < tol
            break
        end
    end
    return x
end

function trust_region_subproblem!(λ, p, q, Δ, B₀, g, maxiters)
    for i in 1:maxiters
        if norm(p) <= Δ
            break
        end
        B = B₀ + λ*I
        F = cholesky(B)
        p = F \ -g
        q = F'\ p
        λ = λ + (p'p)/(q'p) * (norm(p) - Δ)/Δ
    end
    return λ, p
end