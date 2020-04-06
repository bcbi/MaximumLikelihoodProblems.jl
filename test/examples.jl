import Random
import Test

Test.@testset "examples" begin
    Test.@testset "linear regression" begin
        Random.seed!(123)
        include(joinpath(examples_directory,
                         "linear_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "linear_regression.jl"))

    end

    Test.@testset "logistic regression" begin
        Random.seed!(123)
        include(joinpath(examples_directory,
                         "logistic_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "logistic_regression.jl"))
    end

    Test.@testset "multinomial logistic regression" begin
        Random.seed!(123)
        include(joinpath(examples_directory,
                         "multinomial_logistic_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "multinomial_logistic_regression.jl"))
    end
end
