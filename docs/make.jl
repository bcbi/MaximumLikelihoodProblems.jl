import MaximumLikelihoodProblems

import Documenter
import Literate
import Random

_this_filename = @__FILE__
_docs = dirname(_this_filename)
root = dirname(_docs)
docs = joinpath(root, "docs")
docs_src = joinpath(root, "docs", "src")
docs_src_examples = joinpath(root, "docs", "src", "examples")
examples_directory = joinpath(root, "examples")

examples_list = ["linear_regression",
                 "logistic_regression",
                 "multinomial_logistic_regression"]

for example in examples_list
    example_filename = joinpath(examples_directory,
                                "$(example).jl")
    Literate.markdown(example_filename,
                      docs_src_examples)
end

Random.seed!(123)

generated_examples = [joinpath("examples", "$(example).md") for example in examples_list]

Documenter.makedocs(;
    root = docs,
    modules = [MaximumLikelihoodProblems],
    format = Documenter.HTML(),
    pages = vcat(["Home" => "index.md",],
                  generated_examples,
                  ["api_public.md",]),
    repo = "https://github.com/bcbi/MaximumLikelihoodProblems.jl/blob/{commit}{path}#L{line}",
    sitename = "MaximumLikelihoodProblems.jl",
    authors = "Dilum P. Aluthge",
)

Documenter.deploydocs(;
    repo="github.com/bcbi/MaximumLikelihoodProblems.jl",
)
