import LogDensityProblems
import ProgressMeter
import TransformVariables

const default_learning_rate = 1e-4
const default_max_iterations = 1_000_000
const default_show_progress_meter = true
const default_throw_convergence_exception = true
const default_tolerance = 1e-10

"""
    fit(transformed_gradient_problem, theta_hat_initial; kwargs...)

Find the maximum likelihood estimatator for the parameters `theta`.

# Arguments
- `transformed_gradient_problem`
- `theta_hat_initial`

# Keyword Arguments
- `learning_rate`. Default value: $(default_learning_rate)
- `max_iterations`. Default value: $(default_max_iterations)
- `show_progress_meter`. Default value: $(default_show_progress_meter)
- `throw_convergence_exception`. Default value: $(default_throw_convergence_exception)
- `tolerance`. Default value: $(default_tolerance)

See the documentation for fully worked-out examples.
"""
function fit(transformed_gradient_problem,
             theta_hat_initial;
             learning_rate = default_learning_rate,
             max_iterations = default_max_iterations,
             show_progress_meter = default_show_progress_meter,
             throw_convergence_exception = default_throw_convergence_exception,
             tolerance = default_tolerance)
    transformed_log_density = parent(transformed_gradient_problem)
    transformation = transformed_log_density.transformation
    theta_hat_initial_deepcopy = deepcopy(theta_hat_initial)
    theta_hat_initial_deepcopy_inversetransformed = TransformVariables.inverse(transformation,
                                                                               theta_hat_initial_deepcopy)
    theta_hat_initial_deepcopy_inversetransformed_deepcopy = deepcopy(theta_hat_initial_deepcopy_inversetransformed)
    theta_hat_new_inversetransformed = _fit(transformed_gradient_problem,
                                            theta_hat_initial_deepcopy_inversetransformed_deepcopy;
                                            learning_rate = learning_rate,
                                            max_iterations = max_iterations,
                                            show_progress_meter = show_progress_meter,
                                            throw_convergence_exception = throw_convergence_exception,
                                            tolerance = tolerance)
    theta_hat_new = TransformVariables.transform(transformation,
                                                 theta_hat_new_inversetransformed)
    return theta_hat_new
end

function _fit(transformed_gradient_problem,
              theta_hat_initial;
              learning_rate,
              max_iterations,
              show_progress_meter,
              throw_convergence_exception,
              tolerance)
    theta_hat_new = deepcopy(theta_hat_initial)
    if show_progress_meter
        progress = ProgressMeter.ProgressThresh(tolerance, "Maximizing:")
    end
    for iter = 1:max_iterations
        theta_hat_old = theta_hat_new
        log_likelihood_value, gradient = LogDensityProblems.logdensity_and_gradient(transformed_gradient_problem,
                                                                                    theta_hat_old)
        update = learning_rate * gradient
        theta_hat_new = theta_hat_old + update
        update_norm = sum(abs, update)
        if show_progress_meter
            showvalues = [(:iteration, iter),
                          (:loglikelihood, log_likelihood_value)]
            ProgressMeter.update!(progress,
                                  update_norm;
                                  showvalues = showvalues)
        end
        if update_norm < tolerance
            @info("Successfully converged after $(iter) iterations.")
            return theta_hat_new
        end
    end
    if show_progress_meter
        ProgressMeter.finish!(progress)
    end
    if throw_convergence_exception
        @error("Failed to converge after $(max_iterations) iterations.")
        throw(ConvergenceException())
    else
        @warn("Failed to converge after $(max_iterations) iterations.")
    end
    return theta_hat_new
end
