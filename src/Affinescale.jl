
function update_vectors_ak_bk!(ak, bk, x, lb, ub)
    ak .= x .- lb
    bk .= ub .- x
end

function calculate_tk(ak, bk, gk, Delta)
    tk = sqrt(sum(ak.*gk)+sum(bk.*abs.(gk)))/Delta
    return tk
end

function update_Dk!(Dk, tk, ak, bk, gk, Delta, epsilon)
    for i in axis(Dk, 1)
        if (ak[i] <= Delta) & (gk[i] >= epsilon*ak[i])
            Dk[i,i] = tk * sqrt(ak[i]/gk[i])
        elseif (bk[i] <= Delta) & (-gk[i] >= epsilon*bk[i])
            Dk[i,i] = tk * sqrt(bk[i]/abs(gk[i]))
        else
            Dk[i,i] = 1
        end
    end
end
