"""
Affine scaling methods according to Wang and Yuan (2013)
"""



function update_vectors_ak_bk!(ak, bk, x, lb, ub)
    ak .= x .- lb
    bk .= ub .- x
end

function update_Dk!(Dk, ak, bk, gk, Delta, epsilon)
    index_a = findall((ak .<= Delta) .& (gk .>= epsilon.*ak))
    index_b = findall((bk .<= Delta) .& (-gk .>= epsilon.*bk))
    tk = sqrt(sum(ak[index_a].*gk[index_a])+sum(bk[index_b].*abs.(gk[index_b])))/Delta

    for i in axes(Dk, 1)
        if (ak[i] <= Delta) & (gk[i] >= epsilon*ak[i])
            Dk[i,i] = tk * sqrt(ak[i]/gk[i])
        elseif (bk[i] <= Delta) & (-gk[i] >= epsilon*bk[i])
            Dk[i,i] = tk * sqrt(bk[i]/abs(gk[i]))
        else
            Dk[i,i] = 1
        end
    end
end

function update_Dk2!(Dk, ak, bk, Delta, epsilon)
    index_a = findall((ak .<= Delta))
    index_b = findall((bk .<= Delta))
    tk = sqrt(sum(ak[index_a].*gk[index_a])+sum(bk[index_b].*abs.(gk[index_b])))/Delta

    for i in axes(Dk, 1)
        if (ak[i] <= Delta)
            Dk[i,i] = tk * sqrt(ak[i]/gk[i])
        elseif (bk[i] <= Delta) & (-gk[i] >= epsilon*bk[i])
            Dk[i,i] = tk * sqrt(bk[i]/abs(gk[i]))
        else
            Dk[i,i] = 1
        end
    end
end
