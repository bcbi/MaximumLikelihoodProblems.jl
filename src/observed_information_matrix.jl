@inline function observed_information_matrix(d::Derivatives)
    hess_mat = d.hessian_matrix
    result = -1 * _pseudoinverse(hess_mat)
    return result
end

@inline function observed_information_matrix(transformed_gradient_problem,
                                             theta)
    return observed_information_matrix(derivatives(transformed_gradient_problem,
                                                   theta))
end
