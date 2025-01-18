using LinearAlgebra
using Test
include("../src/projected_gradient.jl")

x = [0.5, 0.1, 0.9]
l = -[0.3, 0.3, 0.3]
u = [0.7, 0.7, 0.7]

@test projection(x, l, u) == [0.5, 0.3, 0.7]

G = rand(3,3)
c = rand(3)

g = G*x + c
q(x) = 0.5*x'G*x + c'x
for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,1, 2]
    @test projected_steepest_descent(t, x, g, l, u) ≈ projection(x - t*g, l, u) atol=1e-9
    println(q(projected_steepest_descent(t, x, g, l, u)))
    println(projection(x - t*g, l, u))
end
include("../src/projected_gradient.jl")
cauchy_point(x, c, G, l, u, 5)