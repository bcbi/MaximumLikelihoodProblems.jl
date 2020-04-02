import MaximumLikelihoodProblems

import Documenter
import Literate
import Random

_this_filename = @__FILE__
_docs_directory = dirname(_this_filename)
root_directory = dirname(_docs_directory)
docs_directory = joinpath(root_directory, "docs")
docs_source_directory = joinpath(root_directory, "docs", "src")
examples_directory = joinpath(root_directory, "examples")

examples_list = ["linear_regression",
                 "logistic_regression",
                 "multinomial_logistic_regression"]

for example in examples_list
    example_filename = joinpath(examples_directory, "$(example).jl")
    Literate.markdown(example_filename,
                      docs_source_directory)
end

Random.seed!(123)

Documenter.makedocs(;
    root=docs_directory,
    modules=[MaximumLikelihoodProblems],
    format=Documenter.HTML(),
    pages = vcat(["Home" => "index.md",],
                  ["$(example).md" for example in examples_list],
                  ["api.md",]),
    repo="https://github.com/bcbi/MaximumLikelihoodProblems.jl/blob/{commit}{path}#L{line}",
    sitename="MaximumLikelihoodProblems.jl",
    authors="Dilum P. Aluthge",
    # assets=String[],
)

Documenter.deploydocs(;
    repo="github.com/bcbi/MaximumLikelihoodProblems.jl",
)
