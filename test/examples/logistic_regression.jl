import Statistics
import Test

β_true = [1.0, 2.0]
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
