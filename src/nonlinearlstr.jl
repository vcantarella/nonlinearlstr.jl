module nonlinearlstr
    include("affinescale.jl")
    include("tcg.jl")
    include("qr_nlls.jl")
    include("levenberg-marquardt.jl")
    include("levenberg_marquadt_svd.jl")
    using LinearAlgebra
    include("algorithms.jl")
    include("trf_utils.jl")
end