using Pkg
Pkg.activate("test")
Pkg.develop(path=".")
Pkg.instantiate()
include("test/bounded_tests.jl")
