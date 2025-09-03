# functions to create the affine scale according to coleman and Li.
function affine_scale_matrix(x, lb, ub, g; ϵ = 1e-16)
    v = ones(length(x))
    jᵥ = zeros(length(x))
    for i in eachindex(x)
        if (g[i] < 0) & (ub[i] < Inf)
            v[i] = x[i] - ub[i]
            jᵥ[i] = sign(g[i])
        elseif (g[i] ≥ 0) & (lb[i] > -Inf)
            v[i] = x[i] - lb[i]
            jᵥ[i] = sign(g[i])
        elseif g[i] < 0 #ui = Infs
            v[i] = -1
            jᵥ[i] = 0
        else # g[i] > 0  & li = -Inf
            v[i] = 1
            jᵥ[i] = 0
        end
    end
    D = Diagonal(diagm(1 ./sqrt.(abs.(v).+ϵ)))
    Jᵥ = Diagonal(diagm(g)) * Diagonal(diagm(jᵥ))
    return D, Jᵥ, v
end