abstract type ScalingStrategy end
struct NoScaling <: ScalingStrategy end
struct JacobianScaling <: ScalingStrategy end
struct ColemanandLiScaling <: ScalingStrategy end