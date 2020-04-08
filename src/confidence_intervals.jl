import Distributions

const default_confidence_interval_significance_level = 0.95

function critical_value(level)
    if !(0.5 < level < 1)
        msg = "`level` must be greater than 0.5 and less than 1."
        throw(ArgumentError(msg))
    end
    alpha = 1 - level
    standard_normal_distribution = Distributions.Normal(0, 1)
    critical_probability = alpha/2
    result = -1 * Distributions.quantile(standard_normal_distribution,
                                        critical_probability)
    always_assert(result > 0, "critical value is greater than zero")
    return result
end

function margin_of_error(transformed_gradient_problem,
                         theta;
                         level)
    _critical_value = critical_value(level)
    _stderror_vec = stderror_vector(derivatives(transformed_gradient_problem,
                                                theta))
    _margin_of_error = _critical_value * _stderror_vec
    return _margin_of_error
end

"""
    confint(transformed_gradient_problem, theta; level = 0.95)

# Arguments
- `transformed_gradient_problem`
- `theta`

# Keyword Ar
- `level`. Default value: $(default_confidence_interval_significance_level)


See the documentation for fully worked-out examples.
"""
@inline function confint(transformed_gradient_problem,
                                         theta;
                                         level = default_confidence_interval_significance_level)
    if !(0.5 < level < 1)
        msg = string("`level` must be greater than 0.5 and less than 0.1. ",
                     "For example, for a 95% confidence interval, set `level = 0.95`.",
                     "For a 99% confidence interval, set `level = 0.99`.")
        throw(ArgumentError(msg))
    end
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_inversetransformed = TransformVariables.inverse(transformation,
                                                          theta)
    _margin_of_error = margin_of_error(transformed_gradient_problem,
                                       theta;
                                       level = level)
    lower = theta_inversetransformed - _margin_of_error
    upper = theta_inversetransformed + _margin_of_error
    lower_transformed = TransformVariables.transform(transformation,
                                                     lower)
    upper_transformed = TransformVariables.transform(transformation,
                                                     upper)
    return lower_transformed, upper_transformed
end
