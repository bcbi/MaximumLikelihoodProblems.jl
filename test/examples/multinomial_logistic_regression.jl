import MaximumLikelihoodProblems

import Statistics
import Test

β_true = [1.0 2.0 3.0; 4.0 5.0 6.0]
Test.@test typeof(β_hat) == typeof(β_true)
Test.@test ndims(β_hat) == ndims(β_true)
Test.@test size(β_hat) == size(β_true)
absolute_error = abs.(β_hat - β_true)
square_error = abs2.(β_hat - β_true)
absolute_error_proportional = absolute_error ./ abs.(β_true)
square_error_proportional = square_error ./ abs.(β_true)
Test.@test sum(absolute_error) < 0.60
Test.@test sum(square_error) < 0.06
Test.@test sum(absolute_error_proportional) < 0.35
Test.@test sum(square_error_proportional) < 0.03
Test.@test maximum(absolute_error) < 0.30
Test.@test maximum(square_error) < 0.05
Test.@test maximum(absolute_error_proportional) < 0.10
Test.@test maximum(square_error_proportional) < 0.01
Test.@test Statistics.mean(absolute_error) < 0.95
Test.@test Statistics.mean(square_error) < 0.0095
Test.@test Statistics.mean(absolute_error_proportional) < 0.05
Test.@test Statistics.mean(square_error_proportional) < 0.0055

# coverage for the "failed to converge" code path
β_hat_initial_guess = zeros(size_β)
θ_hat_initial = (; β = β_hat_initial_guess)
θ_hat = MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                      θ_hat_initial;
                                      max_iterations = 10)
@info("The previous warning message (\"Warning: Failed to converge after 10 iterations\") was expected. It is a normal part of the test suite.")
Test.@test_throws MaximumLikelihoodProblems.ConvergenceException MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                                                 θ_hat_initial;
                                                                 max_iterations = 10,
                                                                 throw_convergence_exception = true)
@info("The previous error message (\"Error: Failed to converge after 10 iterations\") was expected. It is a normal part of the test suite.")
