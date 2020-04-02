import MaximumLikelihoodProblems

import Random
import Statistics
import Test

_this_test_filename = @__FILE__
_test_directory = dirname(_this_test_filename)
root_directory = dirname(_test_directory)
examples_directory = joinpath(root_directory, "examples")
test_directory = joinpath(root_directory, "test")

Test.@testset "MaximumLikelihoodProblems.jl" begin
    include("examples.jl")
end
