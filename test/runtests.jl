using nonlinearlstr
using Test

@testset "nonlinearlstr.jl" begin
    using CondaPkg
    CondaPkg.add("scipy")
    using PythonCall
    # Write your tests here.
    include("test_subproblems.jl")
    include("hard_luksan_problems.jl")
    include("problem_testing.jl")
end
