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
function tcg(
    H,
    g::AbstractVector,
    Δ::Real,
    l::AbstractVector,
    u::AbstractVector,
    tol::Real,
    max_iter::Int,
)
    # Step 1: Initialization
    n = length(g)
    d = zeros(n) #Initial guess is zero (Powell, 2009)
    g = g
    v = g
    p = -v
    s = ones(Int, n) # inactive set
    touch_bound = false
    for i = 1:max_iter
        # Step 1: Update active set
        if touch_bound
            println("bound touched")
            for j in eachindex(d)
                if (abs(d[j] - l[j]) < tol && g[j] ≥ 0) ||
                   (abs(d[j] - u[j]) < tol && g[j] ≤ 0)
                    p[s] .= 0
                end
            end
        end
        p[s] .= 0 #inactivating bound constraints
        if sum(p) == 0
            return d # No active constraints, return current solution
        end
        κ = p'H*p
        if κ ≤ 0
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any((d + σ*p) .≤ l)
                σ = minimum((l-d) ./ p)
                touch_bound = true
            elseif any((d + σ*p) .≥ u)
                σ = maximum((u-d) ./ p)
                touch_bound = true
            end
            return d + σ*p
        end
        α = g'v / κ
        if norm(d + α*p) ≥ Δ
            σ = (- d'p + √((d'p)^2 + (p'p)*(Δ^2 - d'd))) / (p'p)
            if any((d + σ*p) .≤ l)
                σ = minimum((l-d) ./ p)
                touch_bound = true
            elseif any((d + σ*p) .≥ u)
                σ = maximum((u-d) ./ p)
                touch_bound = true
            end
            return d + σ*p
        end

        if any((d + α*p) .≤ l)
            α = minimum((l-d) ./ p)
            touch_bound = true
            new_touches = findall((d + α*p) .≤ l)
            valid_touches = [k for k in new_touches if g[k] ≥ 0]
            s = vcat(s, valid_touches)

        elseif any((d + α*p) .≥ u)
            α = maximum((u-d) ./ p)
            touch_bound = true
            new_touches = findall((d + α*p) .≥ u)
            valid_touches = [k for k in new_touches if g[k] ≤ 0]
            s = vcat(s, valid_touches)
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

"""
    tcgnlss(f, J, Δ, l, u, tol, max_iter)

Truncated conjugate gradient method for solving bound-constrained nonlinear least squares trust region subproblems.

This function solves the trust region subproblem for nonlinear least squares:
    minimize    0.5 * ||f + J*d||²
    subject to  ||d|| ≤ Δ
                l ≤ d ≤ u

where the Hessian approximation is J'*J (Gauss-Newton approximation).

# Arguments
- `f::AbstractVector`: The residual vector at the current point
- `J::AbstractMatrix`: The Jacobian matrix of the residual function
- `Δ::Real`: The trust region radius
- `l::AbstractVector`: The lower bounds for the variables
- `u::AbstractVector`: The upper bounds for the variables  
- `tol::Real`: The convergence tolerance for the gradient norm
- `max_iter::Int`: The maximum number of iterations

# Returns
- `d::AbstractVector`: The solution to the trust region subproblem

# Description
This implementation uses the Gauss-Newton approximation H ≈ J'*J for the Hessian,
making it particularly suitable for nonlinear least squares problems. The algorithm
handles bound constraints using an active set strategy similar to the TRSBOX method.

# Example

# References
- Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Chapter 7. The Trust-Region subproblem. Trust region methods. Siam.
- Powell, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization without derivatives. Cambridge NA Report NA2009/06.

"""
function tcgnlss(
    f::AbstractVector,
    J::AbstractMatrix,
    Δ::Real,
    l::AbstractVector,
    u::AbstractVector,
    tol::Real,
    max_iter::Int,
)
    # Step 1: Initialization
    d = zeros(size(J, 2)) #Initial guess is zero Powell(2009)
    g = J'f
    v = g
    p = -v
    s = Int[] # inactive set
    k = 0 #iteration counter
    touch_bound = false
    for i = 1:max_iter
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


using LinearAlgebra

"""
    trsbox!(n, npt, xpt, xopt, gopt, hq, pq, sl, su, delta, xnew, d, gnew, xbdi, s, hs, hred)

This is a Julia translation of the `TRSBOX` subroutine from Powell's BOBYQA Fortran implementation.

It seeks to find a step `d` from `xopt` that approximately minimizes 
    a quadratic model Q within a trust region of radius `delta`,
     subject to lower and upper bounds `sl` and `su`.
     The method uses a truncated conjugate gradient algorithm on the free variables,
     restarting the process if a new variable hits a bound.
     If the trust region boundary is reached,
     it performs a 2D search on the boundary to seek further improvement.

### Arguments (modified in-place)
- `xnew`: On exit, contains the new optimal point `xopt + d`.
- `d`: On exit, the calculated trial step from `xopt`.
- `gnew`: On exit, the gradient of the quadratic model at `xnew`.
- `xbdi`: Workspace vector indicating fixed variables (-1 for lower, 1 for upper, 0 for free).
- `s`, `hs`, `hred`: Workspace vectors.

### Returns
- `dsq`: The squared norm of the final step `d`.
- `crvmin`: The minimum curvature encountered during the CG steps, or 0.0 if the boundary was hit.
"""
function trsbox!(
    n::Int,
    npt::Int,
    xpt::AbstractMatrix,
    xopt::AbstractVector,
    gopt::AbstractVector,
    hq::AbstractVector,
    pq::AbstractVector,
    sl::AbstractVector,
    su::AbstractVector,
    delta::Float64,
    # Workspace arrays (modified in-place)
    xnew::AbstractVector,
    d::AbstractVector,
    gnew::AbstractVector,
    xbdi::AbstractVector,
    s::AbstractVector,
    hs::AbstractVector,
    hred::AbstractVector,
)

    # --- Helper function for Hessian-vector product ---
    # This corresponds to the block at label 210 in the Fortran code.
    # It calculates `hs = H * s`, where H is the Hessian of the quadratic model.
    # The Hessian is stored as a packed symmetric part `hq` and low-rank updates `pq`.
    function hess_vec_prod!(hs_out, s_in)
        fill!(hs_out, 0.0)
        # Handle the packed symmetric part `hq`
        ih = 0
        for j = 1:n
            for i = 1:j
                ih += 1
                if i < j
                    hs_out[j] += hq[ih] * s_in[i]
                end
                hs_out[i] += hq[ih] * s_in[j]
            end
        end
        # Handle the low-rank updates from `pq`
        for k = 1:npt
            if pq[k] != 0.0
                temp = dot(@view(xpt[k, :]), s_in)
                temp *= pq[k]
                hs_out .+= temp .* @view(xpt[k, :])
            end
        end
    end

    # --- Initialization ---
    # The sign of gopt[i] gives the sign of the change to the i-th variable
    # that will reduce Q. xbdi shows whether to fix the i-th variable
    # at a bound initially. nact is the number of fixed variables.
    iterc = 0
    nact = 0
    for i = 1:n
        xbdi[i] = 0.0
        if xopt[i] <= sl[i]
            if gopt[i] >= 0.0
                xbdi[i] = -1.0
            end
        elseif xopt[i] >= su[i]
            if gopt[i] <= 0.0
                xbdi[i] = 1.0
            end
        end
        if xbdi[i] != 0.0
            nact += 1
        end
        d[i] = 0.0
        gnew[i] = gopt[i]
    end

    delsq = delta^2
    qred = 0.0
    crvmin = -1.0

    # Main loop for the solver. It can switch between CG and 2D boundary search.
    # We use a `while` loop to replace the GOTO structure.
    main_loop_active = true
    is_in_cg_phase = true

    while main_loop_active
        # --- Conjugate Gradient (CG) Phase ---
        if is_in_cg_phase
            # This block corresponds to labels 20 and 30 in the Fortran code.
            # It starts/restarts the conjugate gradient method.
            beta = 0.0
            ggsav = 0.0 # Will store the previous gredsq

            # Inner CG loop
            while true
                # Set the search direction. It's steepest descent on the first iteration
                # of a restart (when beta == 0).
                stepsq = 0.0
                for i = 1:n
                    if xbdi[i] != 0.0
                        s[i] = 0.0
                    elseif beta == 0.0
                        s[i] = -gnew[i]
                    else
                        s[i] = beta * s[i] - gnew[i]
                    end
                    stepsq += s[i]^2
                end

                if stepsq == 0.0
                    main_loop_active = false
                    break # Exit inner CG loop, will also exit main loop
                end

                gredsq = 0.0
                if beta == 0.0
                    gredsq = stepsq
                    itermax = iterc + n - nact
                else
                    for i = 1:n
                        if xbdi[i] == 0.0
                            gredsq += gnew[i]^2
                        end
                    end
                end

                # Termination check for small projected gradient
                if gredsq * delsq <= 1.0e-4 * qred^2
                    main_loop_active = false
                    break
                end

                # --- Line Search Calculation ---
                # This block corresponds to label 50.
                hess_vec_prod!(hs, s)

                resid = delsq
                ds = 0.0
                shs = 0.0
                for i = 1:n
                    if xbdi[i] == 0.0
                        resid -= d[i]^2
                        ds += s[i] * d[i]
                        shs += s[i] * hs[i]
                    end
                end

                stplen = 0.0
                blen = 0.0 # Step to trust region boundary
                if resid > 0.0
                    temp = sqrt(stepsq * resid + ds^2)
                    blen = ds < 0.0 ? (temp - ds) / stepsq : resid / (temp + ds)
                    stplen = blen
                end

                if shs > 0.0
                    stplen = min(stplen, gredsq / shs)
                end

                # Reduce stplen if necessary to preserve simple bounds.
                iact = 0
                for i = 1:n
                    if s[i] != 0.0
                        xsum = xopt[i] + d[i]
                        temp = s[i] > 0.0 ? (su[i] - xsum) / s[i] : (sl[i] - xsum) / s[i]
                        if temp < stplen
                            stplen = temp
                            iact = i
                        end
                    end
                end

                # Update curvature, gradient, and step. sdec is the reduction in Q.
                sdec = 0.0
                if stplen > 0.0
                    iterc += 1
                    temp = shs / stepsq
                    if iact == 0 && temp > 0.0
                        crvmin = crvmin == -1.0 ? temp : min(crvmin, temp)
                    end
                    ggsav = gredsq

                    # Update gnew and d
                    gredsq_new = 0.0
                    for i = 1:n
                        gnew[i] += stplen * hs[i]
                        if xbdi[i] == 0.0
                            gredsq_new += gnew[i]^2
                        end
                        d[i] += stplen * s[i]
                    end
                    gredsq = gredsq_new

                    sdec = max(0.0, stplen * (ggsav - 0.5 * stplen * shs))
                    qred += sdec
                end

                # If a new bound was hit, restart the CG method.
                if iact > 0
                    nact += 1
                    xbdi[iact] = s[iact] < 0.0 ? -1.0 : 1.0
                    delsq -= d[iact]^2
                    if delsq <= 0.0
                        crvmin = 0.0
                        is_in_cg_phase = false # Switch to 2D search
                    end
                    break # Exit inner CG loop to restart it
                end

                # If trust region boundary not hit, continue CG or terminate.
                if stplen < blen
                    if iterc == itermax || sdec <= 0.01 * qred
                        main_loop_active = false
                        break
                    end
                    beta = gredsq / ggsav
                    # continue inner CG loop
                else # Hit trust region boundary
                    crvmin = 0.0
                    is_in_cg_phase = false # Switch to 2D search
                    break # Exit inner CG loop
                end
            end # End of inner CG loop
        end # End of CG phase

        # If we broke from an inner check, exit the main loop too.
        if !main_loop_active
            break
        end

        # --- 2D Boundary Search Phase ---
        # Corresponds to labels 100, 120, 150. This part executes if the
        # CG phase hit the trust region boundary (is_in_cg_phase is false).
        if !is_in_cg_phase
            if nact >= n - 1
                break # Not enough free variables for a 2D search
            end

            # Prepare for the alternative iteration (label 100)
            dredsq = 0.0
            dredg = 0.0
            gredsq = 0.0
            for i = 1:n
                s[i] = xbdi[i] == 0.0 ? d[i] : 0.0
                if xbdi[i] == 0.0
                    dredsq += d[i]^2
                    dredg += d[i] * gnew[i]
                    gredsq += gnew[i]^2
                end
            end

            # First Hessian-vector product for the 2D search
            hess_vec_prod!(hred, s) # hred now holds H*d_reduced

            # Let the search direction S be a linear combination of reduced D and G
            # that is orthogonal to reduced D (label 120).
            iterc += 1
            temp = gredsq * dredsq - dredg^2
            if temp <= 1.0e-4 * qred^2
                break # Exit main loop
            end
            temp = sqrt(temp)

            for i = 1:n
                s[i] = xbdi[i] == 0.0 ? (dredg * d[i] - dredsq * gnew[i]) / temp : 0.0
            end
            sredg = -temp

            # Calculate bound on the angle of the alternative iteration (ANGBD)
            angbd = 1.0
            iact = 0
            xsav = 0.0
            for i = 1:n
                if xbdi[i] == 0.0
                    tempa = xopt[i] + d[i] - sl[i]
                    tempb = su[i] - xopt[i] - d[i]
                    if tempa <= 0.0 # Variable is already at a bound, fix it and restart
                        nact += 1
                        xbdi[i] = -1.0
                        is_in_cg_phase = true # Go back to CG phase
                        @goto restart_main_loop # Use goto to simulate Fortran's 'go to 100' logic
                    elseif tempb <= 0.0
                        nact += 1
                        xbdi[i] = 1.0
                        is_in_cg_phase = true
                        @goto restart_main_loop
                    end

                    ssq = d[i]^2 + s[i]^2
                    temp = ssq - (xopt[i] - sl[i])^2
                    if temp > 0.0
                        temp = sqrt(temp) - s[i]
                        if angbd * temp > tempa
                            angbd = tempa / temp
                            iact = i
                            xsav = -1.0
                        end
                    end
                    temp = ssq - (su[i] - xopt[i])^2
                    if temp > 0.0
                        temp = sqrt(temp) + s[i]
                        if angbd * temp > tempb
                            angbd = tempb / temp
                            iact = i
                            xsav = 1.0
                        end
                    end
                end
            end

            # Second Hessian-vector product
            hess_vec_prod!(hs, s) # hs now holds H*s

            # Curvatures for the alternative iteration (label 150)
            shs = 0.0
            dhs = 0.0
            dhd = 0.0
            for i = 1:n
                if xbdi[i] == 0.0
                    shs += s[i] * hs[i]
                    dhs += d[i] * hs[i]
                    dhd += d[i] * hred[i]
                end
            end

            # Seek the greatest reduction in Q over a range of angles
            redmax = 0.0
            isav = 0
            redsav = 0.0
            rdprev = 0.0
            rdnext = 0.0
            iu = round(Int, 17.0 * angbd + 3.1)

            for i = 1:iu
                angt = angbd * i / iu
                sth = 2.0 * angt / (1.0 + angt^2)
                temp = shs + angt * (angt * dhd - 2.0 * dhs)
                rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp)
                if rednew > redmax
                    redmax = rednew
                    isav = i
                    rdprev = redsav
                elseif i == isav + 1
                    rdnext = rednew
                end
                redsav = rednew
            end

            # If no reduction, exit. Otherwise, refine angle and update.
            if isav == 0
                break
            end

            angt = angbd * isav / iu
            if isav < iu
                temp = (rdnext - rdprev) / (2.0 * redmax - rdprev - rdnext)
                angt = angbd * (isav + 0.5 * temp) / iu
            end

            cth = (1.0 - angt^2) / (1.0 + angt^2)
            sth = 2.0 * angt / (1.0 + angt^2)
            temp = shs + angt * (angt * dhd - 2.0 * dhs)
            sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp)

            if sdec <= 0.0
                break
            end

            # Update gnew, d for the 2D step
            qred += sdec
            for i = 1:n
                gnew[i] += (cth - 1.0) * hred[i] + sth * hs[i]
                if xbdi[i] == 0.0
                    d[i] = cth * d[i] + sth * s[i]
                end
            end

            # If the angle was restricted by a bound, fix that variable and restart.
            if iact > 0 && isav == iu
                nact += 1
                xbdi[iact] = xsav
                is_in_cg_phase = true # Go back to CG phase
                continue # Restart main loop
            end

            # If reduction is too small, exit. Otherwise, loop for another 2D search.
            if sdec <= 0.01 * qred
                break
            end

            # This 'continue' simulates the 'go to 120' to try another 2D step
            is_in_cg_phase = false
            @label restart_main_loop
            # The loop will naturally continue here unless a break was hit.
        end
    end # End of main loop

    # --- Finalization (label 190) ---
    # Set xnew to xopt + d, carefully respecting the bounds.
    dsq = 0.0
    for i = 1:n
        xnew[i] = min(su[i], max(sl[i], xopt[i] + d[i]))
        if xbdi[i] == -1.0
            xnew[i] = sl[i]
        elseif xbdi[i] == 1.0
            xnew[i] = su[i]
        end
        d[i] = xnew[i] - xopt[i]
        dsq += d[i]^2
    end

    return dsq, crvmin
end
