import MaximumLikelihoodProblems

import Statistics
import Test

this_test_filename = @__FILE__
test_directory = dirname(this_test_filename)
root_directory = dirname(test_directory)
examples_directory = joinpath(root_directory, "examples")

Test.@testset "MaximumLikelihoodProblems.jl" begin
    include("examples.jl")
end
