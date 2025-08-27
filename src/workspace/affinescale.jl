"""
    update_Dk!(affine_cache, u, lb, ub, gk, Δ, ϵ)

Bound constrained least squares may be improved by affine scaling methods.
Here we implement an update for the affine scaling matrix `Dk` and its inverse `inv_Dk` from the paper:
    A trust region method based on a new affine scaling technique for simple bounded optimization. Wang & Yuan, 2013.
TODO: test the affine scaling benefit for common problems.
Update the affine scaling matrix `Dk` and its inverse `inv_Dk` based on the current guess and bounds.

# Arguments
- `affine_cache`: A named tuple containing the affine scaling matrix `Dk`, its inverse `inv_Dk`, and the vectors `ak` and `bk`.
- `u::Vector{Float64}`: The current point.
- `lb::Vector{Float64}`: The lower bounds.
- `ub::Vector{Float64}`: The upper bounds.
- `gk::Vector{Float64}`: The gradient vector.
- `Δ::Float64`: The trust region radius.
- `ϵ::Float64`: A small positive number to ensure numerical stability.

# Description
This function updates the affine scaling matrix `Dk` and its inverse `inv_Dk` based on the subject vectors `ak`, `bk`, and `gk`. The update is performed by calculating a scaling factor `tk` and adjusting the diagonal elements of `Dk` and `inv_Dk` accordingly.
"""
function update_Dk!(affine_cache, u, lb, ub, gk, Δ, ϵ)
    (; Dk, inv_Dk, ak, bk) = affine_cache
    
    # Pre-allocate for the numerator of tk
    tk_numerator_sq = 0.0
    
    # Store indices to avoid re-computation
    index_map = zeros(Int, length(u)) # 0 for neither, 1 for S_k^1, 2 for S_k^2

    min_distance = 1e-12
    for i in eachindex(u)
        ak[i] = max(u[i] - lb[i], min_distance)
        bk[i] = max(ub[i] - u[i], min_distance)

        if (ak[i] <= Δ) && (gk[i] >= ϵ * ak[i])
            tk_numerator_sq += ak[i] * gk[i]
            index_map[i] = 1
        elseif (bk[i] <= Δ) && (-gk[i] >= ϵ * bk[i])
            tk_numerator_sq += bk[i] * abs(gk[i])
            index_map[i] = 2
        end
    end

    tk = sqrt(tk_numerator_sq) / Δ

    for i in axes(Dk, 1)
        if index_map[i] == 1
            Dk[i, i] = tk * sqrt(ak[i] / gk[i])
        elseif index_map[i] == 2
            Dk[i, i] = tk * sqrt(bk[i] / abs(gk[i]))
        else
            Dk[i, i] = 1.0
        end
        inv_Dk[i, i] = 1.0 / Dk[i, i]
    end
end