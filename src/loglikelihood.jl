import LogDensityProblems
import TransformVariables

"""
    loglikelihood(transformed_gradient_problem, theta)

Return the value of the log likelihood function evaluated at `theta`.

# Arguments
- `transformed_gradient_problem`
- `theta`
"""
function loglikelihood(transformed_gradient_problem,
                       theta)
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_inversetransformed = TransformVariables.inverse(transformation,
                                                          theta)
    result = _loglikelihood(transformed_gradient_problem,
                            theta_inversetransformed)
    return result
end

function _loglikelihood(transformed_gradient_problem,
                        theta)
    log_likelihood_value, gradient = LogDensityProblems.logdensity_and_gradient(transformed_gradient_problem,
                                                                                theta)
    return log_likelihood_value
end
