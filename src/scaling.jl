abstract type ScalingStrategy end
struct NoScaling <: ScalingStrategy end
struct JacobianScaling <: ScalingStrategy end
struct ColemanandLiScaling <: ScalingStrategy end

function scaling(::NoScaling, J::AbstractMatrix; kwargs...)
    return I(size(J, 2))
end

function scaling(::JacobianScaling, J; τ = 1e-12, kwargs...)
    n = size(J, 2)
    Dk = Diagonal(ones(eltype(J), n))
    column_norms = [norm(J[:, i]) for i in axes(J, 2)]
    for i in axes(J, 2)
        Dk[i, i] = max(τ, column_norms[i])
    end
    return Dk
end