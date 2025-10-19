function compute_cl_scaling(x, g, lb, ub)
    """Compute Coleman-Li scaling vector and its derivatives."""
    n = length(x)
    v = ones(n)
    dv = zeros(n)
    for i = 1:n
        if g[i] < 0 && ub[i] < Inf
            v[i] = ub[i] - x[i]
            dv[i] = -1
        elseif g[i] > 0 && lb[i] > -Inf
            v[i] = x[i] - lb[i]
            dv[i] = 1
        end
    end
    # Ensure minimum distance to avoid numerical issues
    min_distance = 1e-10
    for i = 1:n
        if dv[i] != 0 && v[i] < min_distance
            v[i] = min_distance
        end
    end
    return v, dv
end

"""
    function step_size_to_bound(x, s, lb, ub)
    
    Compute a min_step size required to reach a bound.
    The function computes a positive scalar t, such that x + s * t is on
    the bound.
    Arguments
    ---------
    x : Vector{Float64}
        Current point.
    s : Vector{Float64}
        Search direction.
    lb : Vector{Float64}
        Lower bounds.
    ub : Vector{Float64}
        Upper bounds.
    Returns
    -------
    step : Float64
        Computed step. Non-negative value.
    hits : Vector{Int}
        Each element indicates whether a corresponding variable reaches the
        bound:
                *  0 - the bound was not hit.
                * -1 - the lower bound was hit.
                *  1 - the upper bound was hit.
"""
function step_size_to_bound(x, s, lb, ub)

    non_zero = findall(s .!= 0)
    s_non_zero = s[non_zero]
    steps = fill(Inf, size(x))

    # Calculate steps to bounds for non-zero components
    steps[non_zero] = max.(
        (lb[non_zero] - x[non_zero]) ./ s_non_zero,
        (ub[non_zero] - x[non_zero]) ./ s_non_zero,
    )

    # Replace negative steps with Inf (they won't contribute to the minimum)
    steps[steps .< 0] .= Inf

    # Find the minimum step
    min_step = minimum(steps)

    # Determine which bounds are hit
    hits = (steps .== min_step) .* Int.(sign.(s))

    return min_step, hits
end

function select_step(x, J, diag_h, g, p, p_scaled, d, Δ, lb, ub, theta)
    """Select the best step according to Trust Region Reflective algorithm."""

    # Check if the full step is within bounds
    if all(lb .<= x + p .<= ub)
        p_value = 0.5 * norm(J * p)^2 + dot(g, p)
        return p, p_scaled, p_value
    end

    # Find distance to boundary
    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction
    r_scaled = copy(p_scaled)
    r_scaled[hits .!= 0] .*= -1
    r = d .* r_scaled

    # Restrict trust-region step to hit the bound
    p *= p_stride
    p_scaled *= p_stride
    x_on_bound = x + p

    # Compute reflected step parameters
    # This requires intersect_trust_region and other helper functions
    # ...

    # Compute anti-gradient direction
    ag_scaled = -g .* d
    ag = d .* ag_scaled

    # Compare all steps and select the best
    # ...

    # For now, let's just return the scaled-back step
    return p, p_scaled, 0.0
end
