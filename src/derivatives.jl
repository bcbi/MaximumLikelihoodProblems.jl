import ForwardDiff # TODO: remove the dependency on ForwardDiff.jl
import LogDensityProblems
import TransformVariables

@inline function _compute_hessian(transformed_gradient_problem,
                                  theta_inversetransformed)
    @assert transformed_gradient_problem isa LogDensityProblems.ForwardDiffLogDensity
    ℓ = transformed_gradient_problem.ℓ
    logdensity_closure = LogDensityProblems._logdensity_closure(ℓ)
    hess_mat = ForwardDiff.hessian(logdensity_closure,
                                   theta_inversetransformed)
    return hess_mat
end

@inline function derivatives(transformed_gradient_problem,
                             theta)
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_inversetransformed = TransformVariables.inverse(transformation,
                                                          theta)
    log_likelihood_val, grad_vec = LogDensityProblems.logdensity_and_gradient(transformed_gradient_problem,
                                                                              theta_inversetransformed)
    hess_mat = _compute_hessian(transformed_gradient_problem,
                                theta_inversetransformed)
    result = Derivatives(transformation,
                         log_likelihood_val,
                         grad_vec,
                         hess_mat)
    return result
end

@inline function gradient(d::Derivatives)
    transformation = d.transformation
    grad_vec = d.gradient_vector
    grad = TransformVariables.transform(transformation,
                                        grad_vec)
    return grad
end

@inline function gradient(transformed_gradient_problem,
                          theta)
    return gradient(derivatives(transformed_gradient_problem,
                                theta))
end

@inline function gradient_vector(d::Derivatives)
    return d.gradient_vector
end

@inline function gradient_vector(transformed_gradient_problem,
                                 theta)
    return gradient_vector(derivatives(transformed_gradient_problem,
                                       theta))
end

@inline function hessian_matrix(d::Derivatives)
    return d.hessian_matrix
end

@inline function hessian_matrix(transformed_gradient_problem,
                               theta)
    return hessian_matrix(derivatives(transformed_gradient_problem,
                                      theta))
end
