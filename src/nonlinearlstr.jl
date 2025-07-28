module nonlinearlstr
    include("affinescale.jl")
    include("tcg.jl")
    include("qr_nlls.jl")
    include("levenberg-marquardt.jl")
    using LinearAlgebra
    include("algorithms.jl")
end
