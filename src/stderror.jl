@inline function stderror_vector(m::AbstractMatrix)
    diagonal_entries = _get_diagonal_entries(m)
    result = sqrt.(diagonal_entries)
    return result
end

@inline function stderror_vector(d::Derivatives)
    return stderror_vector(observed_information_matrix(d))
end

@inline function stderror_vector(transformed_gradient_problem,
                                 theta)
    return stderror_vector(derivatives(transformed_gradient_problem,
                                       theta))
end

@inline function stderror(d::Derivatives)
    transformation = d.transformation
    stderror_vec = stderror_vector(d)
    result = TransformVariables.transform(transformation,
                                          stderror_vec)
    return result
end

"""
    stderror(transformed_gradient_problem, theta)

# Arguments
- `transformed_gradient_problem`
- `theta`

See the documentation for fully worked-out examples.
"""
@inline function stderror(transformed_gradient_problem,
                          theta)
    return stderror(derivatives(transformed_gradient_problem,
                                theta))
end
