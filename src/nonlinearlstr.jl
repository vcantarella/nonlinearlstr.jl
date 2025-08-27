module nonlinearlstr
    include("levenberg-marquardt.jl")
    include("levenberg_marquadt_svd.jl")
    using LinearAlgebra
    include("algorithms.jl")
    include("bounded_algorithms.jl")
end