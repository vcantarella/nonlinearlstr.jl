using CUTEst
using NLPModels
using Enlsip
using NLSProblems
using JSOSolvers
using PRIMA
using NonlinearSolve
using Pkg, Revise
using DataFrames, CSV, CairoMakie
using LeastSquaresOptim
using LinearAlgebra, Statistics

Pkg.develop(PackageSpec(path="/Users/vcantarella/.julia/dev/nonlinearlstr"))
using nonlinearlstr

function find_bounded_problems(max_vars=50)
    """Find bounded NLS problems from NLSProblems.jl package"""
    # Get all available NLS problems
    all_problems = setdiff(names(NLSProblems), [:NLSProblems])
    
    valid_problems = []
    
    for prob_name in all_problems
            prob = eval(prob_name)()
            
            # Filter by size and check if it's a valid NLS problem
            if !bound_constrained(prob)
                println("  Problem $prob_name is not bound constrained, skipping")
                finalize(prob)
                continue
            elseif prob.meta.nvar <= max_vars #&& 
               #prob.meta.nequ > 0 &&  # Has residuals
               isa(prob, AbstractNLSModel)
                
                push!(valid_problems, prob_name)
                finalize(prob)
            else
                finalize(prob)
            end
    end
    
    println("Found $(length(valid_problems)) valid bounded NLS problems (≤ $max_vars variables)")
    return valid_problems
end


function create_nls_functions(prob)
    """Create Julia function wrappers for NLSProblems problem"""
    
    # Get problem infor
    #m = prob.meta.nequ  # Number of residuals
    x0 = copy(prob.meta.x0)
    bl = copy(prob.meta.lvar)
    bu = copy(prob.meta.uvar)
    
    # Define functions using NLPModels interface
    residual_func(x) = residual(prob, x)
    jacobian_func(x) = Matrix(jac_residual(prob, x))
    n,m = size(jacobian_func(x0))
    
    # Create objective as 0.5 * ||r||²
    obj_func(x) = obj(prob, x)
    grad_func(x) = grad(prob, x)
    
    # Use Gauss-Newton approximation for Hessian
    hess_func(x) = begin
        J = jacobian_func(x)
        return J' * J
    end
    
    return (
        n=n, m=m, x0=x0, bl=bl, bu=bu,
        residual_func=residual_func,
        jacobian_func=jacobian_func,
        obj_func=obj_func,
        grad_func=grad_func,
        hess_func=hess_func,
        name=prob.meta.name
    )
end

probs = find_bounded_problems()

nls_functions = [create_nls_functions(eval(prob_name)()) for prob_name in probs]

prob_data = nls_functions[1]  # Use the first problem for testing
result = nonlinearlstr.bounded_gauss_newton(
                prob_data.residual_func, prob_data.jacobian_func, 
                prob_data.x0, prob_data.bl, prob_data.bu,;
                max_iter=100, gtol=1e-8
            )
x_opt, r_opt, g_opt, iterations = result
final_obj = 0.5 * dot(r_opt, r_opt)
converged = norm(g_opt, 2) < 1e-6    