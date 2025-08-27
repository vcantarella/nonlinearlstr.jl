# functions to create the affine scale according to coleman and Li.
function affine_scale_matrix(x, lb, ub, g; ϵ = 1e-16)
    v = ones(length(x))
    for i in eachindex(x)
        if (g[i] < 0) & (ub[i] < Inf)
            v[i] = x[i] - ub[i]
        elseif (g[i] ≥ 0) & (lb[i] > -Inf)
            v[i] = x[i] - lb[i]
        elseif g[i] < 0 #ui = Infs
            v[i] = -1
        else # g[i] > 0  & li = -Inf
            v[i] = 1
        end
    end
    D = Diagonal(diagm(1 ./sqrt.(abs.(v))))
    jᵥ = zeros(length(x))
    for i in eachindex(g)
        if abs(g[i]) < ϵ
            jᵥ[i] = 0
        elseif (lb[i] == -Inf) || (ub[i] == Inf)
            jᵥ[i] = 0
        else
            jᵥ[i] = sign(g[i])
        end
    end
    Jᵥ = Diagonal(diagm(g)) * Diagonal(diagm(jᵥ))
    return D, Jᵥ
end