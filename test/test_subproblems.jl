using Test
using LinearAlgebra
using Statistics
using Random
using ForwardDiff
using BenchmarkTools
using Pkg
using Revise
using NonlinearSolve
using nonlinearlstr

@testset "Unconstrained Subproblem Tests" begin

    # Setup test data - used across all test sets
    Random.seed!(123)  # For reproducible tests
    m, n = 10, 4
    J = randn(m, n)
    f_true = randn(n)
    p_true = randn(n)
    f = -J * p_true + 0.1 * randn(m)  # Add some noise
    λ = 0.8
    
    # Helper function for reference solution
    function get_p(J, f, λ)
        n = size(J, 2)
        J_aug = [J; sqrt(λ) * I(n)]
        b = [-f; zeros(n)]
        return J_aug \ b
    end
    
    p_ref = get_p(J, f, λ)
    size_p = norm(p_ref)

    @testset "Cache Construction" begin
        @testset "QR Cache" begin
            strat = nonlinearlstr.QRSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
            
            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <: Union{LinearAlgebra.QR, LinearAlgebra.QRCompactWY}
        end
        
        @testset "SVD Cache" begin
            strat = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
            
            @test cache.scaling_matrix == I(n)
            @test typeof(cache.factorization) <: LinearAlgebra.SVD
        end
    end

    @testset "Low-level Solving Functions" begin
        @testset "SVD Augmented System Solver" begin
            strat = nonlinearlstr.SVDSolve()
            svdls = svd(J)
            p_svd = nonlinearlstr.solve_augmented(strat, svdls, J, Diagonal(ones(n)), -f, λ)
            
            @test p_svd ≈ p_ref rtol=1e-10
        end
        
        @testset "Derivative dp/dλ - QR Method" begin
            strat = nonlinearlstr.QRSolve()
            J_aug = [J; sqrt(λ) * I(n)]
            qrf = qr(J_aug)
            Dk = I(n)
            
            # Analytical derivative
            dp_dλ = nonlinearlstr.solve_for_dp_dlambda_scaled(strat, qrf, p_ref, Dk)
            
            # Finite difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)
            
            @test dp_dλ ≈ dp_dλ_fd atol=1e-6
        end
        
        @testset "Derivative dp/dλ - SVD Method" begin
            strat = nonlinearlstr.SVDSolve()
            F = svd(J)
            Dk = I(n)
            
            # Analytical derivative
            dp_dλ_svd = nonlinearlstr.solve_for_dp_dlambda_scaled(strat, F, Dk, λ, -f)
            
            # Finite difference reference
            dp_dλ_fd = ForwardDiff.derivative(λ -> get_p(J, f, λ), λ)
            
            @test dp_dλ_svd ≈ dp_dλ_fd atol=1e-6
        end
    end

    @testset "Trust Region Subproblem Solver" begin
        @testset "QR-based Solver" begin
            strat = nonlinearlstr.QRSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
            
            λ_qr, δ_qr = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)
            
            @test abs(λ_qr - λ) < 1e-3
            @test norm(δ_qr) ≈ size_p atol=1e-3
            @test δ_qr ≈ p_ref atol=1e-3
        end
        
        @testset "SVD-based Solver" begin
            strat = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()
            cache = nonlinearlstr.SubproblemCache(strat, scaling, J)
            
            λ_svd, δ_svd = nonlinearlstr.solve_subproblem(strat, J, f, size_p, cache)
            
            @test abs(λ_svd - λ) < 1e-3
            @test norm(δ_svd) ≈ size_p atol=1e-3
            @test δ_svd ≈ p_ref atol=1e-3
        end
        
        @testset "Method Consistency" begin
            # Both methods should give similar results
            strat_qr = nonlinearlstr.QRSolve()
            strat_svd = nonlinearlstr.SVDSolve()
            scaling = nonlinearlstr.NoScaling()
            
            cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
            cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)
            
            λ_qr, δ_qr = nonlinearlstr.solve_subproblem(strat_qr, J, f, size_p, cache_qr)
            λ_svd, δ_svd = nonlinearlstr.solve_subproblem(strat_svd, J, f, size_p, cache_svd)
            
            @test λ_qr ≈ λ_svd atol=1e-2
            @test δ_qr ≈ δ_svd atol=1e-2
        end
    end

    @testset "Performance Benchmarks" begin
        # Note: These are for development/profiling, not automated testing
        println("\n=== Performance Benchmarks ===")
        
        println("Benchmarking QR-based solve:")
        strat_qr = nonlinearlstr.QRSolve()
        scaling = nonlinearlstr.NoScaling()
        cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
        @benchmark nonlinearlstr.solve_subproblem($strat_qr, $J, $f, $size_p, $cache_qr)
        
        println("Benchmarking SVD-based solve:")
        strat_svd = nonlinearlstr.SVDSolve()
        cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)
        @benchmark nonlinearlstr.solve_subproblem($strat_svd, $J, $f, $size_p, $cache_svd)
    end

    @testset "Type Stability" begin
        # Check for type instabilities
        println("\n=== Type Stability Analysis ===")
        
        println("QR method type analysis:")
        strat_qr = nonlinearlstr.QRSolve()
        scaling = nonlinearlstr.NoScaling()
        cache_qr = nonlinearlstr.SubproblemCache(strat_qr, scaling, J)
        @code_warntype nonlinearlstr.solve_subproblem(strat_qr, J, f, size_p, cache_qr)
        
        println("SVD method type analysis:")
        strat_svd = nonlinearlstr.SVDSolve()
        cache_svd = nonlinearlstr.SubproblemCache(strat_svd, scaling, J)
        @code_warntype nonlinearlstr.solve_subproblem(strat_svd, J, f, size_p, cache_svd)
    end

end
