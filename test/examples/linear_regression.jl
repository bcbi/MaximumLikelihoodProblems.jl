import MaximumLikelihoodProblems

import GLM
import Statistics
import Test

σ_true = 0.5
β_true = [1.0, 2.0, -1.0]

# @show σ_true
# @show σ_hat
# @show β_true
# @show β_hat

Test.@test typeof(σ_hat) == typeof(σ_true)
Test.@test σ_hat isa Real
Test.@test σ_true isa Real
Test.@test isfinite(σ_hat)
Test.@test σ_hat > 0.1
Test.@test abs(σ_hat - σ_true) < 0.03
Test.@test abs2(σ_hat - σ_true) < 1e-3

Test.@test typeof(β_hat) == typeof(β_true)
Test.@test ndims(β_hat) == ndims(β_true)
Test.@test size(β_hat) == size(β_true)
absolute_error = abs.(β_hat - β_true)
square_error = abs2.(β_hat - β_true)
absolute_error_proportional = absolute_error ./ abs.(β_true)
square_error_proportional = square_error ./ abs.(β_true)
Test.@test sum(absolute_error) < 0.030
Test.@test sum(square_error) < 1e-3
Test.@test sum(absolute_error_proportional) < 0.030
Test.@test sum(square_error_proportional) < 1e-3
Test.@test maximum(absolute_error) < 0.013
Test.@test maximum(square_error) < 1e-3
Test.@test maximum(absolute_error_proportional) < 0.013
Test.@test maximum(square_error_proportional) < 1e-3
Test.@test Statistics.mean(absolute_error) < 0.010
Test.@test Statistics.mean(square_error) < 1e-3
Test.@test Statistics.mean(absolute_error_proportional) < 0.007
Test.@test Statistics.mean(square_error_proportional) < 1e-3

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

n, p = size(X)
beta_hat_ols = ( X' * X )\( X' * y )
beta_hat_mle = beta_hat_ols
y_hat_ols = X * beta_hat_ols
epsilon_hat_ols = y - y_hat_ols
S_ols = epsilon_hat_ols' * epsilon_hat_ols
sigma_2_hat_ols = (S_ols) / (n - p) # OLS estimator for σ²
sigma_2_hat_mle = (n - p) / (n) * sigma_2_hat_ols # MLE estimator for σ²
sigma_hat_ols = sqrt(sigma_2_hat_ols)
sigma_hat_mle = sqrt(sigma_2_hat_mle)
Test.@test isapprox(σ_hat, sigma_hat_ols; atol = 1e-4)
Test.@test isapprox(σ_hat, sigma_hat_mle; atol = 1e-4)
Test.@test isapprox(β_hat, beta_hat_ols)
Test.@test isapprox(β_hat, beta_hat_mle)

external_model_glm = GLM.lm(X, y)
sigma_hat_external_model_glm = GLM.dispersion(external_model_glm)
Test.@test isapprox(σ_hat, sigma_hat_external_model_glm; atol = 1e-4)
beta_hat_external_model_glm = GLM.coef(external_model_glm)
Test.@test isapprox(β_hat, beta_hat_external_model_glm)
