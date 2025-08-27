using nonlinearlstr
using Test

@testset "nonlinearlstr.jl" begin
    # Write your tests here.
    include("test_lambda.jl")
    include("test_svd.jl")
end
