module Internal

import ForwardDiff # TODO: remove the dependency on ForwardDiff.jl
import LogDensityProblems
import LinearAlgebra
import TransformVariables

const _default_fuzz_factor = 0.01

function gradient_vector(transformed_gradient_problem,
                         theta)
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_inversetransformed = TransformVariables.inverse(transformation,
                                                          theta)
    result = _gradient_vector(transformed_gradient_problem,
                              theta_inversetransformed)
    return result
end

function _gradient_vector(transformed_gradient_problem,
                          theta)
    log_likelihood_value, gradient = LogDensityProblems.logdensity_and_gradient(transformed_gradient_problem,
                                                                                theta)
    return gradient
end

function hessian_matrix(transformed_gradient_problem,
                        theta)
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_inversetransformed = TransformVariables.inverse(transformation,
                                                          theta)
    result = _hessian_matrix(transformed_gradient_problem,
                             theta_inversetransformed)
    return result
end

function _hessian_matrix(transformed_gradient_problem,
                         theta)
    @assert transformed_gradient_problem isa LogDensityProblems.ForwardDiffLogDensity
    ℓ = transformed_gradient_problem.ℓ
    logdensity_closure = LogDensityProblems._logdensity_closure(ℓ)
    hessian = ForwardDiff.hessian(logdensity_closure,
                                  theta)
    return hessian
end

function _is_approximately_zero(a::AbstractArray;
                                tolerance = 1e-4)
    result = all( abs.(a) .< tolerance )
    return result
end

function _is_approximately_hermitian(m::AbstractMatrix;
                                     tolerance = 1e-4)
    LinearAlgebra.checksquare(m)
    result = _is_approximately_zero(m - LinearAlgebra.adjoint(m);
                                    tolerance = tolerance)
    return result
end

function _all_eigenvalues_are_negative(m::AbstractMatrix;
                                       fuzz_factor = _default_fuzz_factor)
    eigvals = LinearAlgebra.eigvals(m)
    a = all( eigvals .< 0 )
    b = all( eigvals .< -abs(fuzz_factor) )
    result = a && b
    return result
end

function _all_eigenvalues_are_positive(m::AbstractMatrix;
                                       fuzz_factor = _default_fuzz_factor)
    eigvals = LinearAlgebra.eigvals(m)
    a = all( eigvals .> 0 )
    b = all( eigvals .> abs(fuzz_factor) )
    result = a && b
    return result
end

function _is_approximately_negative_definite(m::AbstractMatrix;
                                             tolerance = 1e-4,
                                             fuzz_factor = _default_fuzz_factor)
    # Let `m` be a Hermitian matrix. Then `m` is negative
    # definite if and only if all of its eigenvalues are
    # negative.
    a = _is_approximately_hermitian(m;
                                    tolerance = tolerance)
    b = _all_eigenvalues_are_negative(m;
                                      fuzz_factor = fuzz_factor)
    result = a && b
    return result
end

function _is_approximately_positive_definite(m::AbstractMatrix;
                                             tolerance = 1e-4,
                                             fuzz_factor = _default_fuzz_factor)
    # Let `m` be a Hermitian matrix. Then `m` is positive
    # definite if and only if all of its eigenvalues are
    # positive.
    a = _is_approximately_hermitian(m;
                                    tolerance = tolerance)
    b = _all_eigenvalues_are_positive(m;
                                      fuzz_factor = fuzz_factor)
    result = a && b
    return result
end

end # module
