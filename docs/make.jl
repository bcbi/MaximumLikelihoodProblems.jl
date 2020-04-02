using Documenter, MaximumLikelihoodProblems

makedocs(;
    modules=[MaximumLikelihoodProblems],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/bcbi/MaximumLikelihoodProblems.jl/blob/{commit}{path}#L{line}",
    sitename="MaximumLikelihoodProblems.jl",
    authors="Dilum Aluthge",
    assets=String[],
)

deploydocs(;
    repo="github.com/bcbi/MaximumLikelihoodProblems.jl",
)
