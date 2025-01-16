using LinearAlgebra
"""
    cg(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector, tol::Real, max_iter::Int)

Solve the linear system `Ax = b` using the Conjugate Gradient method.

# Arguments
- `A::AbstractMatrix`: A symmetric positive definite matrix.
- `b::AbstractVector`: A vector.
- `x0::AbstractVector`: Initial guess for the solution.
- `tol::Real`: Tolerance for the stopping criterion.
- `max_iter::Int`: Maximum number of iterations.

# Returns
- `x::AbstractVector`: The approximate solution to the system `Ax = b`.

# Description
The Conjugate Gradient method is an iterative algorithm for solving systems of linear equations with a symmetric positive definite matrix. The quadratic form minimized is:

    q(x) = 0.5 * x' * A * x + b' * x + c

where `A` is a symmetric positive definite matrix, `b` is a vector, and `c` is a scalar.

# Example
```julia
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
x0 = [2.0, 1.0]
tol = 1e-6
max_iter = 1000
x = cg(A, b, x0, tol, max_iter)
```
"""
function cg(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector, tol::Real, max_iter::Int)
    x = x0
    g = A*x + b #g_0 initial gradient
    u = -g #u_0 initial search direction is the steepest decent
    k = 0 #iteration counter
    for i in 1:max_iter
        λꜝ = - (g' * u) / (u' * A * u) #step size
        x = x + λꜝ * u #update x
        g = A*x + b #update gradient
        if norm(g) < tol
            return x
        end
        β = (g' * A * u) / (u' * A * u) #update search direction
        u = -g + β * u #the search direction is a linear combination of the steepest decent and the previous search direction
        k = k + 1
    end
    return x
end


"""
    tcg0(cache, H, g, d, Δ, l, u, tol, max_iter)
    Truncated trust region conjugate gradient method to solve the trust region subproblem.
    based on Steinhaug-Toint algorithm and the TRSBOX.
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
"""
function tcg(H::AbstractMatrix, g::AbstractVector,Δ::Real,
             l::AbstractVector, u::AbstractVector,
             tol::Real, max_iter::Int)
    # Step 1: Initialization
    n = length(g)
    d = zeros(n) #Initial guess is zero Powell(2009)
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
        Β = (g_1'v_1) / (g'v)
        p = -v_1 + Β*p
        g = g_1
        if norm(g) < tol
            return d
        end
        v = v_1
    end
    return d
end

mutable struct TCGCache{T}
    # Solution vectors
    d::Vector{T}    # current solution
    g::Vector{T}    # gradient
    v::Vector{T}    # residual
    p::Vector{T}    # search direction
    Hp::Vector{T}   # matrix-vector product
    
    # Computation buffers
    pHp::T          # p'Hp value
    gp::T           # g'p value  
    vv::T           # v'v value
    vv_next::T      # next v'v value
    alpha::T        # step length
    beta::T         # conjugate direction parameter
    
    # Sets
    s::Vector{Int}  # inactive set
end

function TCGCache(n::Int, T::DataType=Float64)
    TCGCache(
        zeros(T, n),  # d
        zeros(T, n),  # g 
        zeros(T, n),  # v
        zeros(T, n),  # p
        zeros(T, n),  # Hp
        zero(T),      # pHp
        zero(T),      # gp
        zero(T),      # vv
        zero(T),      # vv_next
        zero(T),      # alpha
        zero(T),      # beta
        Int[]         # s
    )
end

function tcg!(cache::TCGCache, H::AbstractMatrix, g::AbstractVector, 
             Δ::Real, l::AbstractVector, u::AbstractVector,
             tol::Real, max_iter::Int)
    fill!(cache.d, 0)
    copyto!(cache.g, g)
    copyto!(cache.v, g)
    copyto!(cache.p, -g)
    empty!(cache.s)
    
    k = 0
    touch_bound = false
    
    for i in 1:max_iter
        # Update active set
        if touch_bound
            empty!(cache.s)
            for j in eachindex(cache.d)
                if (abs(cache.d[j] - l[j]) < tol && cache.g[j] >= 0) || 
                   (abs(cache.d[j] - u[j]) < tol && cache.g[j] <= 0)
                    push!(cache.s, j)
                end
            end
        end
        
        # Inactivate bound constraints
        cache.p[cache.s] .= 0
        
        # Compute κ = p'Hp
        mul!(cache.Hp, H, cache.p)
        cache.pHp = dot(cache.p, cache.Hp)
        
        # Check curvature
        if cache.pHp <= 0
            σ = (-dot(cache.d, cache.p) + 
                 sqrt(dot(cache.d, cache.p)^2 + 
                      dot(cache.p, cache.p)*(Δ^2 - dot(cache.d, cache.d)))) / 
                 dot(cache.p, cache.p)
            
            if any(cache.d + σ*cache.p .<= l)
                σ = minimum((l .- cache.d) ./ cache.p)
            elseif any(cache.d + σ*cache.p .>= u)
                σ = maximum((u .- cache.d) ./ cache.p)
            end
            
            @. cache.d += σ * cache.p
            return cache.d
        end
        
        # Compute step length
        cache.alpha = dot(cache.g, cache.v) / cache.pHp
        
        # Check trust region
        if norm(cache.d + cache.alpha*cache.p) >= Δ
            σ = (-dot(cache.d, cache.p) + 
                 sqrt(dot(cache.d, cache.p)^2 + 
                      dot(cache.p, cache.p)*(Δ^2 - dot(cache.d, cache.d)))) / 
                 dot(cache.p, cache.p)
            @. cache.d += σ * cache.p
            return cache.d
        end
        
        # Update solution
        @. cache.d += cache.alpha * cache.p
        @. cache.v += cache.alpha * cache.Hp
        
        # Check bounds
        if any(cache.d .<= l) || any(cache.d .>= u)
            touch_bound = true
        end
        
        # Update gradient
        cache.vv = dot(cache.v, cache.v)
        if sqrt(cache.vv) < tol
            return cache.d
        end
        
        # Update search direction
        cache.beta = cache.vv / dot(cache.g, cache.v)
        copyto!(cache.g, cache.v)
        @. cache.p = -cache.v + cache.beta * cache.p
        
        k += 1
    end
    
    return cache.d
end

using BenchmarkTools

# Test problem setup
n = 100
H = rand(n,n)
H = H'H  # make symmetric
g = rand(n)
Δ = 1.0
l = fill(-1.0, n)
u = fill(1.0, n)
tol = 1e-6
max_iter = 1000

# Benchmark original
@btime tcg($H, $g, $Δ, $l, $u, $tol, $max_iter)

# Benchmark with cache 
cache = TCGCache(n)
@btime tcg!($cache, $H, $g, $Δ, $l, $u, $tol, $max_iter)

d = tcg(H, g, Δ, l, u, tol, max_iter)
d_cache = tcg!(cache, H, g, Δ, l, u, tol, max_iter)