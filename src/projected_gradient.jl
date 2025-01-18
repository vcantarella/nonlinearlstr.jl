using LinearAlgebra

"""
    projection(x::Vector{T}, l::Vector{T}, u::Vector{T}) where T<:Real

Projects a vector `x` onto the feasible region defined by the lower bounds `l` and upper bounds `u`.

# Arguments
- `x::Vector{T}`: The input vector to be projected.
- `l::Vector{T}`: The vector of lower bounds.
- `u::Vector{T}`: The vector of upper bounds.

# Returns
- `Vector{T}`: The projected vector that lies within the bounds defined by `l` and `u`.

# Description
This function implements the projection of a vector onto the feasible region defined
     by the lower and upper bounds. Each element of the input vector `x` is projected
     to lie within the corresponding elements of `l` and `u`. The algorithm follows 
     the method described in "Numerical Optimization" by Jorge Nocedal and Stephen Wright.
"""
# Example
function projection(x, l, u)
    return max.(min.(x, u), l)
end

"""
    projected_steepest_descent(t, x, g, l, u)
"""
function projected_steepest_descent(t, x, g, l, u)
    return projection(x - t*g, l, u)
end

"""
    cauchy_point(x, c, G, l, u, Δ)

    Calculate the Cauchy point based on bound constraints and a trust region.

    # Arguments
    - `x::Vector{Float64}`: The current iterate.
    - `c::Vector{Float64}`: The linear term of the quadratic model.
    - `G::Matrix{Float64}`: The Hessian matrix of the quadratic model.
    - `l::Vector{Float64}`: The lower bounds of the variables.
    - `u::Vector{Float64}`: The upper bounds of the variables.
    - `Δ::Float64`: The trust region radius.

    # Returns
    - `pC::Vector{Float64}`: The Cauchy point.
    - `active_set::Vector{Int}`: The active set of constraints.

    # Description
    This function computes the Cauchy point, which is an approximation
         to the solution of the trust region subproblem.
    The implementation is based on the method described in
     "Numerical Optimization" by Nocedal and Wright.
"""
function cauchy_point(x, c, G, l, u, Δ)
    g = G*x + c
    # determining the index of the active and inactive constraints
    t = zeros(length(x)) # initialize the vector of step lengths
    for i in eachindex(x)# Defining the steps to each bound
        if g[i] > 0 && l[i] > -Inf
            t[i] = (x[i] - l[i])/g[i]
        elseif g[i] < 0 && u[i] < Inf
            t[i] = (x[i] - u[i])/g[i]
        else
            t[i] = Inf
        end
    end
    # sort the step lengths in ascending order
    sorted_index = sortperm(t)
    t = t[sorted_index]
    # clip t by Δ
    t = [t[i] for i in eachindex(t) if t[i] ≤ Δ]
    # we want the index to build the active set later on:
    sorted_index = sorted_index[1:length(t)]
    # now we loop to each interval between two consective points from t: [t_l, t_f]
    t_l = 0 #initialize the lower bound of the interval
    t_f = 0 #initialize the upper bound of the interval
    active_set = [] #initialize the active set
    for j in eachindex(t)
        p = -g
        inds = t .≤  t[j]
        p[sorted_index[inds]] .= 0 # set the inactive constraints to zero
        # now we get a new quadratic model within the interval [t_l, t_f] (p. 488 Nocedal and Wright)
        # we just need the first and second derivatives of the new quadratic model:
        f_jp = c'p + x'G*p
        f_jpp = p'G'p

        if f_jp > 0 # Then the optimal step is in the lower bound
            t_f = t_l
            break
        else
            Δt = -f_jp/f_jpp
            if 0 < Δt < t[j] - t_l # Then the optimal step is in the interval
                t_f = t_l + Δt
                x = x + Δt*p
                break
            else # Then the optimal step is in the next interval
                Δt = t[j] - t_l
            end
            x = x + Δt*p
        end
        t_l = t[j] # update the lower bound of the interval
        push!(active_set, sorted_index[j]) # add the active constraint to the active set
        if j == length(t) # if we reach the last interval we need to check if we consider the trust radius or not
            if j < length(x)
                # then there is at least one boundary not reached and we must check the trust region boundary
                Δt = -f_jp/f_jpp
                if 0 < Δt < Δ - t_l
                    t_f = t_l + Δt
                else
                    Δt = Δ - t_l
                    t_f = Δ
                end
                x = x + Δt*p
            else # all constraints are active and we are at bounds
                t_f = t[j]
            end
        end
    end
    return projected_steepest_descent(t_f, x, g, l, u), active_set
end

function compute_Z_matrix(diagonal_entries::Vector{T}) where T<:Number
    n = length(diagonal_entries)
    # Find indices of zero entries
    zero_indices = findall(x -> x == 0, diagonal_entries)
    k = length(zero_indices)
    
    # Initialize Z with zeros
    Z = zeros(T, n, k)
    
    # Fill in 1's at appropriate positions
    for (col, zero_idx) in enumerate(zero_indices)
        Z[zero_idx, col] = 1
    end
    
    return Z
end

function projected_cg(c, G, l, u, Δ, maxiters = 100)
    x = zeros(length(c))
    xᶜ, active_set = cauchy_point(x, c, G, l, u, Δ)
    x[active_set] .= xᶜ[active_set]
    P = Diagonal([1 for i in x if i ∉ active_set])
    r = G*x + c
    g = P*r
    d = -g
    touch_bound = false
    for i in 1:maxiters
        κ = d'H*d
        if κ ≤ 0
            return xᶜ
        end
        α = r'g/κ
        if any(x + α*d .≤ l)
            α = minimum((l-x) ./ d)
            touch_bound = true
        elseif any(x + α*d .≥ u)
            α = maximum((u-x) ./ d)
            touch_bound = true
        end
        if norm(x + α*d) ≥ Δ
            σ = (- x'd + √((x'd)^2 + (d'd)*(Δ^2 - x'x))) / (d'd)
            if any(x + σ*d .≤ l)
                σ = minimum((l-x) ./ d)
            elseif any(x + σ*d .≥ u)
                σ = maximum((u-x) ./ d)
            end
            return x + σ*d
        end
        x = x + α*d
        r⁺ = r + α*G*d
        g⁺ = P*r⁺
        β = (r⁺'g⁺) / (r'g)
    end

end