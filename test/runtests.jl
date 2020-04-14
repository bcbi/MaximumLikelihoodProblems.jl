import MaximumLikelihoodProblems

import LinearAlgebra
import Random
import Statistics
import Test

_this_test_filename = @__FILE__
_test_directory = dirname(_this_test_filename)
root_directory = dirname(_test_directory)
examples_directory = joinpath(root_directory, "examples")
test_directory = joinpath(root_directory, "test")

Test.@testset "MaximumLikelihoodProblems.jl" begin
    Test.@testset "linear_algebra.jl" begin
        Test.@testset "_pseudoinverse" begin
            Test.@test MaximumLikelihoodProblems._pseudoinverse([1 2; 3 4]) == LinearAlgebra.inv([1 2; 3 4])
            Test.@test MaximumLikelihoodProblems._pseudoinverse([0 0; 0 0]) == [0 0; 0 0]
        end
    end
    
    Test.@testset "examples" begin
        include("examples.jl")
    end
end
