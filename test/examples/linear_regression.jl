import Statistics
import Test

σ_true = 0.5
β_true = [1.0, 2.0, -1.0]

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
