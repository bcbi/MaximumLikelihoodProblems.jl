import GLM
import LinearAlgebra
import Statistics
import Test

β_true = [1.0, 2.0]

# @show β_true
# @show β_hat

Test.@test typeof(β_hat) == typeof(β_true)
Test.@test ndims(β_hat) == ndims(β_true)
Test.@test size(β_hat) == size(β_true)
absolute_error = abs.(β_hat - β_true)
square_error = abs2.(β_hat - β_true)
absolute_error_proportional = absolute_error ./ abs.(β_true)
square_error_proportional = square_error ./ abs.(β_true)
Test.@test sum(absolute_error) < 0.150
Test.@test sum(square_error) < 0.010
Test.@test sum(absolute_error_proportional) < 0.100
Test.@test sum(square_error_proportional) < 0.010
Test.@test maximum(absolute_error) < 0.100
Test.@test maximum(square_error) < 0.010
Test.@test maximum(absolute_error_proportional) < 0.060
Test.@test maximum(square_error_proportional) < 0.005
Test.@test Statistics.mean(absolute_error) < 0.070
Test.@test Statistics.mean(square_error) < 0.007
Test.@test Statistics.mean(absolute_error_proportional) < 0.050
Test.@test Statistics.mean(square_error_proportional) < 0.010
external_model_glm = GLM.glm(X, y, GLM.Binomial(), GLM.LogitLink())
beta_hat_external_model_glm = GLM.coef(external_model_glm)
Test.@test isapprox(β_hat, beta_hat_external_model_glm)

gradient_vector_at_θ_hat = MaximumLikelihoodProblems.gradient_vector(transformed_gradient_problem,
                                                                     θ_hat)
Test.@test all(abs.(gradient_vector_at_θ_hat) .< 1e-4)
hessian_matrix_at_θ_hat = MaximumLikelihoodProblems.hessian_matrix(transformed_gradient_problem,
                                                                   θ_hat)
Test.@test all(abs.(LinearAlgebra.adjoint(hessian_matrix_at_θ_hat) - hessian_matrix_at_θ_hat) .< 1e-10)
Test.@test all(LinearAlgebra.eigvals(hessian_matrix_at_θ_hat) .< 0)
Test.@test all(LinearAlgebra.eigvals(hessian_matrix_at_θ_hat) .< 100)
