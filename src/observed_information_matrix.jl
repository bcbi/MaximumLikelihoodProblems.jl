@inline function observed_information_matrix(d::Derivatives)
    hess_mat = d.hessian_matrix
    return -hess_mat
end

@inline function observed_information_matrix(transformed_gradient_problem,
                                             theta)
    return observed_information_matrix(derivatives(transformed_gradient_problem,
                                                   theta))
end
