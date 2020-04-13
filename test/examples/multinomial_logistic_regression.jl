import MaximumLikelihoodProblems

import CategoricalArrays
import DataFrames
import Econometrics
import Statistics
import Test

β_true = [1.0 2.0 3.0; 4.0 5.0 6.0]

# @show β_true
# @show β_hat

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
Test.@test maximum(absolute_error_proportional) < 0.20
Test.@test maximum(square_error_proportional) < 0.03
Test.@test Statistics.mean(absolute_error) < 0.95
Test.@test Statistics.mean(square_error) < 0.0095
Test.@test Statistics.mean(absolute_error_proportional) < 0.05
Test.@test Statistics.mean(square_error_proportional) < 0.0055

gradient_vector_at_θ_hat = MaximumLikelihoodProblems.Internal.gradient_vector(transformed_gradient_problem,
                                                                              θ_hat)
hessian_matrix_at_θ_hat = MaximumLikelihoodProblems.Internal.hessian_matrix(transformed_gradient_problem,
                                                                            θ_hat)
Test.@test MaximumLikelihoodProblems.Internal._is_approximately_zero(gradient_vector_at_θ_hat)
Test.@test MaximumLikelihoodProblems.Internal._is_approximately_hermitian(hessian_matrix_at_θ_hat)
Test.@test MaximumLikelihoodProblems.Internal._is_approximately_negative_definite(hessian_matrix_at_θ_hat;
                                                                                  fuzz_factor = 10)
Test.@test MaximumLikelihoodProblems.Internal._is_approximately_positive_definite(-hessian_matrix_at_θ_hat;
                                                                                  fuzz_factor = 10)

external_df = DataFrames.DataFrame()
external_df[!, :X_2] = X[:, 2]
n = size(X, 1)
external_y_string = Vector{String}(undef, n)
for i = 1:n
    external_y_string[i] = string(argmax(y[i, :]))
end
external_y_string_categorical = CategoricalArrays.CategoricalArray(external_y_string;
                                                                   ordered = false)
external_df[!, :y] = external_y_string_categorical
external_formula = Econometrics.@formula(y ~ 1 + X_2)
# external_model_econometrics = Econometrics.fit(Econometrics.EconometricModel,
                                               # external_formula,
                                               # external_df)
# external_model_coef = Econometrics.coef(external_model_econometrics)
# beta_hat_external_model_econometrics = reshape(external_model_coef, (2, 3))
# Test.@test all(isapprox.(β_hat, beta_hat_external_model_econometrics; atol = 1e-2))
# external_model_absolute_error = abs.(β_hat - beta_hat_external_model_econometrics)
# external_model_square_error = abs2.(β_hat - beta_hat_external_model_econometrics)
# external_model_absolute_error_proportional = external_model_absolute_error ./ abs.(beta_hat_external_model_econometrics)
# external_model_square_error_proportional = external_model_square_error ./ abs.(beta_hat_external_model_econometrics)
# Test.@test sum(external_model_absolute_error) < 0.07
# Test.@test sum(external_model_square_error) < 0.001
# Test.@test sum(external_model_absolute_error_proportional) < 0.04
# Test.@test sum(external_model_square_error_proportional) < 1e-4
# Test.@test maximum(external_model_absolute_error) < 0.01
# Test.@test maximum(external_model_square_error) < 1e-4
# Test.@test maximum(absolute_error_proportional) < 0.1
# Test.@test maximum(external_model_square_error_proportional) < 1e-4
# Test.@test Statistics.mean(external_model_absolute_error) < 0.01
# Test.@test Statistics.mean(external_model_square_error) < 1e-4
# Test.@test Statistics.mean(external_model_absolute_error_proportional) < 0.01
# Test.@test Statistics.mean(external_model_square_error_proportional) < 1e-4

# coverage for the "failed to converge" code paths
β_hat_initial_guess = zeros(size_β)
θ_hat_initial = (; β = β_hat_initial_guess)
θ_hat = MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                      θ_hat_initial;
                                      max_iterations = 10,
                                      throw_convergence_exception = false)
@info("The previous warning message (\"Warning: Failed to converge after 10 iterations\") was expected. It is a normal part of the test suite.")
Test.@test_throws(MaximumLikelihoodProblems.ConvergenceException,
                  MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                                θ_hat_initial;
                                                max_iterations = 10,
                                                throw_convergence_exception = true))
@info("The previous error message (\"Error: Failed to converge after 10 iterations\") was expected. It is a normal part of the test suite.")
Test.@test_throws(MaximumLikelihoodProblems.ConvergenceException,
                  MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                                θ_hat_initial;
                                                max_iterations = 10))
@info("The previous error message (\"Error: Failed to converge after 10 iterations\") was expected. It is a normal part of the test suite.")
