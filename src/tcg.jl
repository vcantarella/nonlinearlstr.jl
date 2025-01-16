"""
    tcg(H, g, d, Δ, l, u, tol, max_iter)
    Truncated trust region conjugate gradient method to solve the trust region subproblem.
    based on Steinhaug-Toint algorithm (Conn et al, 2000), with the active set update for bounds based in the TRSBOX (Powell, 2009).
    Checking its capability to solve the trust region subproblem.
    
# Arguments
- `H::AbstractMatrix`: The problem Hessian, or a matrix that approximates the Hessian.
- `g::AbstractVector`: The gradient of the problem.
- `Δ::Real`: The trust region radius.
- `l::AbstractVector`: The lower bound of the bounds constraints.
- `u::AbstractVector`: The upper bound of the bounds constraints.
- `tol::Real`: The tolerance for the convergence of the subproblem.
- `max_iter::Int`: The maximum number of iterations for the subproblem.

# Returns
- `d::AbstractVector`: The solution to the trust region subproblem.`

# Description
We want to solve the subproblem:

    q(d) = fx + 0.5 * d' * H * d + g' * d

where `H` is the problem Hessian, `g` is the gradient of the problem, `d` is the search direction, `Δ` is the trust region radius, `l` is the lower bound of the bounds constraints, `u` is the upper bound of the bounds constraints, `tol` is the tolerance for the convergence of the subproblem, and `max_iter` is the maximum number of iterations for the subproblem.

# Example
```julia
H = [4.0 1.0; 1.0 3.0]
g = [1.0, 2.0]
d = [0.0, 0.0]
Δ = 1.0
l = [-1.0, -1.0]
u = [1.0, 1.0]
tol = 1e-6
max_iter = 1000
d = tcg(H, g, d, Δ, l, u, tol, max_iter)

# References
- Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Chapter 7. The Trust-Region subproblem. Trust region methods. Siam.
- Powell, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives. Cambridge NA Report NA2009/06.
"""
function tcg(H::AbstractMatrix, g::AbstractVector,Δ::Real,
             l::AbstractVector, u::AbstractVector,
             tol::Real, max_iter::Int)
    # Step 1: Initialization
    n = length(g)
    d = zeros(n) #Initial guess is zero (Powell, 2009)
    g = g
    v = g
    p = -v
    s = Int[] # inactive set
    k = 0 #iteration counter
    touch_bound = false
    for i in 1:max_iter
        # Step 1: Update active set
        if touch_bound
            for j in eachindex(d)
                if (abs(d[j] - l[j]) < tol && g[j] ≥ 0) || 
                (abs(d[j] - u[j]) < tol && g[j] ≤ 0)
                    push!(s, j)
                end
            end
        end
        p[s] .= 0 #inactivating bound constraints
        κ = p'H*p
        if κ ≤ 0
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any(d + σ*p .≤ l)
                σ = minimum((l-d) ./ p)
            elseif any(d + σ*p .≥ u)
                σ = maximum((u-d) ./ p)
            end
            return d + σ*p
        end
        α = g'v / κ
        if norm(d + α*p) ≥ Δ
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any(d + σ*p .≤ l)
                σ = minimum((l-d) ./ p)
            elseif any(d + σ*p .≥ u)
                σ = maximum((u-d) ./ p)
            end
            return d + σ*p
        end
        
        if any(d + α*p .≤ l)
            α = minimum((l-d) ./ p)
            touch_bound = true
        elseif any(d + α*p .≥ u)
            α = maximum((u-d) ./ p)
            touch_bound = true
        end
        d = d + α*p
        g_1 = g + α*H*p
        v_1 = g_1
        β = (g_1'v_1) / (g'v)
        p = -v_1 + β*p
        g = g_1
        if norm(g) < tol
            return d
        end
        v = v_1
    end
    return d
end


function tcgnlss(f::AbstractVector, J::AbstractMatrix, Δ::Real,
    l::AbstractVector, u::AbstractVector,
    tol::Real, max_iter::Int)
    # Step 1: Initialization
    d = zeros(size(J,2)) #Initial guess is zero Powell(2009)
    g = J'f
    v = g
    p = -v
    s = Int[] # inactive set
    k = 0 #iteration counter
    touch_bound = false
    for i in 1:max_iter
        # Step 1: Update active set
        if touch_bound
            for j in eachindex(d)
                if (abs(d[j] - l[j]) < tol && g[j] ≥ 0) || 
                (abs(d[j] - u[j]) < tol && g[j] ≤ 0)
                    push!(s, j)
                end
            end
        end
        p[s] .= 0 #inactivating bound constraints
        κ = p'*(J'*(J*p))
        if κ ≤ 0
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any(d + σ*p .≤ l)
                σ = minimum((l-d) ./ p)
            elseif any(d + σ*p .≥ u)
                σ = maximum((u-d) ./ p)
            end
            return d + σ*p
        end
        α = g'v / κ
        if norm(d + α*p) ≥ Δ
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any(d + σ*p .≤ l)
                σ = minimum((l-d) ./ p)
            elseif any(d + σ*p .≥ u)
                σ = maximum((u-d) ./ p)
            end
            return d + σ*p
        end
        
        if any(d + α*p .≤ l)
            α = minimum((l-d) ./ p)
            touch_bound = true
        elseif any(d + α*p .≥ u)
            α = maximum((u-d) ./ p)
            touch_bound = true
        end
        d = d + α*p
        g_1 = g + α*(J'*(J*p))
        v_1 = g_1
        β = (g_1'v_1) / (g'v)
        p = -v_1 + β*p
        g = g_1
        if norm(g) < tol
            return d
        end
        v = v_1
    end
    return d
end