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

function qr_regularized_solve(J::AbstractMatrix, b::AbstractVector, λ::Real)
    m, n = size(J)
    
    # Form augmented system: [J; √λ I][x] = [b; 0]
    if λ > 0
        sqrt_λ = √λ
        # Augmented matrix is [J; √λ I]
        J_aug = [J; sqrt_λ * I(n)]
        b_aug = [b; zeros(n)]
    else
        J_aug = J
        b_aug = b
    end
    
    # QR factorization of augmented system
    #qr_aug = qr(J_aug)
    
    # Solve the system
    x = J_aug \ b_aug
    return x
end

function find_λ!(λ₀, Δ, J, f, maxiters)
    λ = λ₀
    p = zeros(size(J, 2))
    for i in 1:maxiters
        p = qr_regularized_solve(J, -f, λ)
        if norm(p) <= Δ
            break
        end
        #p = F \ -g
        QR = qr(J)
        q = QR.R' \ p
        #q = F'\ p
        λ = λ + (p'p)/(q'q) * (norm(p) - Δ)/Δ
    end
    println("Final step norm: ", norm(p))
    return λ
end


function find_λ2!(Δ, J, f, maxiters)
    l₀ = 0.0
    u₀ = norm(J'f)/Δ
    λ₀ = maximum([1e-3*u₀,√(l₀*u₀)])
    λ = λ₀
    uₖ = u₀
    lₖ = l₀
    p = zeros(size(J, 2))
    for i in 1:maxiters
        if (λ <= lₖ) || (λ > uₖ)
            λ = maximum([1e-3*uₖ,√(lₖ*uₖ)])
        end
        p = qr_regularized_solve(J, -f, λ)
        ϕ = norm(p)-Δ
        if ϕ < 0
            uₖ = λ
        end
        #p = F \ -g
        QR = qr(J, ColumnNorm())
        q = QR.P' * (QR.R' \ p)
        ϕ¹ = -(q'q)/norm(p)
        lₖ = maximum([lₖ, λ-ϕ/ϕ¹])
        #q = F'\ p
        λ = λ - (ϕ+Δ)/Δ*(ϕ/ϕ¹)
    end
    println("Final step norm: ", norm(p))
    return λ, p
end

function smallest_eigenvalue_JTJ(J::AbstractMatrix; shift=1e-8, max_iter=20)
    """Approximate smallest eigenvalue of J^T J using shifted inverse iteration"""
    n = size(J, 2)
    v = randn(n)
    v = v / norm(v)
    
    # Form J^T J + shift*I for numerical stability
    JTJ = J' * J
    JTJ_shifted = JTJ + shift * I(n)
    
    try
        lu_factor = lu(JTJ_shifted)
        
        for i in 1:max_iter
            v_new = lu_factor \ v
            v_new = v_new / norm(v_new)
            
            # Rayleigh quotient for original matrix
            λ = v_new' * JTJ * v_new
            v = v_new
        end
        
        return λ
    catch
        # Fallback: use pseudo-inverse approach
        return minimum(svdvals(J))^2
    end
end

