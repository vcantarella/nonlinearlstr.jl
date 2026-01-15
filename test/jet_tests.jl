using JET
using nonlinearlstr
using Test

@testset "JET checks" begin
    test_package(nonlinearlstr; target_defined_modules = true)
end
