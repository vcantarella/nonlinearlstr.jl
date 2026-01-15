using Test
using BenchmarkTools
using LinearAlgebra
using nonlinearlstr
using StaticArrays

@testset "Solver Allocation Tests" begin
    # Setup a small problem
    n = 10
    m = 20
    x0 = ones(n)

    function res(x)
        r = zeros(m)
        r[1:n] .= x .- 1.0
        return r
    end

    function jac(x)
        J = zeros(m, n)
        J[1:n, 1:n] .= I(n)
        return J
    end

    # Pre-calculate data to test inner functions specifically
    x = copy(x0)
    f = res(x)
    J = jac(x)
    radius = 1.0

    # Test strategies
    strategies = [
        (nonlinearlstr.QRSolve(), nonlinearlstr.NoScaling(), "LM-QR"),
        (nonlinearlstr.SVDSolve(), nonlinearlstr.NoScaling(), "LM-SVD"),
        (nonlinearlstr.QRrecursiveSolve(), nonlinearlstr.NoScaling(), "LM-QR-Recursive"),
    ]

    for (strat, scaling, name) in strategies
        println("\nTesting allocations for $name...")

        # 1. Test Cache Construction
        cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

        # 2. Test solve_subproblem allocations
        allocs = @ballocated nonlinearlstr.solve_subproblem($strat, $J, $f, $radius, $cache)

        println("  solve_subproblem allocations: $allocs bytes")
    end
end

@testset "find_lambda Allocation Tests" begin
    # Setup a small problem
    n = 10
    m = 20
    x0 = ones(n)

    function res(x)
        r = zeros(m)
        r[1:n] .= x .- 1.0
        return r
    end

    function jac(x)
        J = zeros(m, n)
        J[1:n, 1:n] .= I(n)
        return J
    end

    # Pre-calculate data to test inner functions specifically
    x = copy(x0)
    f = res(x)
    J = jac(x)
    radius = 1.0

    # Test strategies
    strategies = [
        (nonlinearlstr.QRSolve(), nonlinearlstr.NoScaling(), "LM-QR"),
        (nonlinearlstr.SVDSolve(), nonlinearlstr.NoScaling(), "LM-SVD"),
        (nonlinearlstr.QRrecursiveSolve(), nonlinearlstr.NoScaling(), "LM-QR-Recursive"),
    ]

    for (strat, scaling, name) in strategies
        println("\nTesting allocations for λ in $name...")

        # 1. Test Cache Construction
        cache = nonlinearlstr.SubproblemCache(strat, scaling, J)

        # 2. Test solve_subproblem allocations find_λ_scaled(strategy, cache, radius, J, Dk, f, 200, 1e-6)
        allocs = @ballocated nonlinearlstr.find_λ_scaled(
            $strat,
            $cache,
            $radius,
            $J,
            $cache.scaling_matrix,
            $f,
            200,
            1e-6,
        )

        println("  solve_subproblem allocations: $allocs bytes")
    end
end
