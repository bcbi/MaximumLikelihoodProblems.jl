import Test

Test.@testset "examples" begin
    Test.@testset "linear regression" begin
        include(joinpath(examples_directory,
                         "linear_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "linear_regression.jl"))
    end

    Test.@testset "logistic regression" begin
        include(joinpath(examples_directory,
                         "logistic_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "logistic_regression.jl"))
    end

    Test.@testset "multinomial logistic regression" begin
        include(joinpath(examples_directory,
                         "multinomial_logistic_regression.jl"))
        include(joinpath(test_directory,
                         "examples",
                         "multinomial_logistic_regression.jl"))
    end
end
