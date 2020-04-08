struct AlwaysAssertionError <: Exception
    msg::String
end

struct ConvergenceException <: Exception
end

struct Derivatives{T, F1, F2, F3}
    transformation::T
    log_likelihood_value::F1
    gradient_vector::Vector{F2}
    hessian_matrix::Matrix{F3}
end
