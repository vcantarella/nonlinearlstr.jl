using nonlinearlstr
using Test
using PythonCall
using CondaPkg
# CondaPkg.add("numpy")
# CondaPkg.add("scipy")

@testset "nonlinearlstr.jl" begin
    include("jet_tests.jl")

    include("test_subproblems.jl")
    include("test_colemanli.jl")
    include("problem_testing_simp.jl")
    include("bounded_tests.jl")
end
