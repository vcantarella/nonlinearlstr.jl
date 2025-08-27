using LinearAlgebra

J = randn(10, 5)

Q, R = qr(J)
λ = 0.1
n = size(J, 2)
p = size(J, 1)
D = √λ*I(n)
R_aug = [R; zeros(p - n, n);D]

Qt = I(p+n)
# Loop from the bottom-right up
for i in n:-1:1
    # 1. Eliminate the diagonal element from the D block
    #    Use row 'i' of R to zero out the element at (p+i, i)
    G_diag, r = givens(R_aug, i, p + i, i)
    R_aug = G_diag * R_aug
    Qt = G_diag * Qt # Accumulate the transform

    # 2. Clean up the "fill-in" created by the above rotation.
    #    Fill-in occurs in row p+i, at columns j = i+1, ..., n
    for j in (i + 1):n
        # Use row 'j' of R to zero out the fill-in at (p+i, j)
        G_fill, r_fill = givens(R_aug, j, p + i, j)
        R_aug = G_fill * R_aug
        Qt = G_fill * Qt # Accumulate the transform
    end
end


function augmented_qr(qrJ, λ)
    Q = qrJ.Q
    R = qrJ.R
    n = size(R, 2)
    m = size(Q, 1)
    D = √λ * I(n)
    if m > n # most of the cases
        R_aug = [R; zeros(m - n, n); D]
    else
        R_aug = [R; D]
    end
    p = m
    #R_aug = [R; zeros(size(D, 1), size(R, 2)); D]
    Qt = I(m+n)
    #p = ifelse(m < n, n, m-n)  # Number of rows in R
    # Loop from the bottom-right up
    for i in n:-1:1
        # 1. Eliminate the diagonal element from the D block
        #    Use row 'i' of R to zero out the element at (p+i, i)
        G_diag, r = givens(R_aug, i, p + i, i)
        R_aug = G_diag * R_aug
        Qt = G_diag * Qt # Accumulate the transform

        # 2. Clean up the "fill-in" created by the above rotation.
        #    Fill-in occurs in row p+i, at columns j = i+1, ..., n
        for j in (i + 1):n
            # Use row 'j' of R to zero out the fill-in at (p+i, j)
            G_fill, r_fill = givens(R_aug, j, p + i, j)
            R_aug = G_fill * R_aug
            Qt = G_fill * Qt # Accumulate the transform
        end
    end
    Q_aug = [Q zeros(m,n); zeros(n,m) I(n)] * Qt'
    return (Q_aug, R_aug)
end

Q_aug, R_augv2 = augmented_qr(qr(J), λ)

J = randn(6, 10)
qru = qr(J)
R = qru.R
Q_aug, R_augv2 = augmented_qr(qru, λ) # This fails! The method won't work with a under determined system
