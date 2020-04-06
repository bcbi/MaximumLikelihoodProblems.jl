# # Linear regression

import MaximumLikelihoodProblems

import Distributions
import ForwardDiff
import LogDensityProblems
import Parameters
import TransformVariables

struct LinearRegression{Ty, TX}
    y::Ty
    X::TX
end

function (problem::LinearRegression)(θ)
    Parameters.@unpack y, X = problem
    Parameters.@unpack β, σ = θ
    η = X*β
    μ = η
    ε = y - μ

    ## these two lines are equivalent:
    ## log_likelihood = Distributions.loglikelihood(Distributions.Normal(0, σ), ε)
    ## log_likelihood = sum(Distributions.logpdf.(Distributions.Normal(0, σ), ε))

    log_likelihood = sum(Distributions.logpdf.(Distributions.Normal(0, σ), ε))
    return log_likelihood
end

N = 10_000

## X has three columns
## the first column (the column of all ones) is the intercept
## the second column is the first covariate
## the third column is the second covariate
X = hcat(ones(N), randn(N), randn(N))

size_β = (3,)
β_true = [1.0, 2.0, -1.0]
σ_true = 0.5
η_true = X * β_true
μ_true = η_true
ε_true = randn(N) .* σ_true
y = μ_true + ε_true

function generate_problem_transformation(p::LinearRegression)
    return TransformVariables.as((β = TransformVariables.as(Array, size(p.X, 2)),
                                  σ = TransformVariables.asℝ₊))
end

problem = LinearRegression(y, X)
transformation = generate_problem_transformation(problem)
transformed_problem = LogDensityProblems.TransformedLogDensity(transformation,
                                                               problem)
transformed_gradient_problem = LogDensityProblems.ADgradient(:ForwardDiff,
                                                             transformed_problem)

β_hat_initial_guess = zeros(size_β)
σ_hat_initial_guess = 1.0

θ_hat_initial = (; β = β_hat_initial_guess,
                   σ = σ_hat_initial_guess)

# θ_hat:

θ_hat = MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                      θ_hat_initial;
                                      learning_rate = 1e-6)

# β_hat:

β_hat = θ_hat[:β]

# σ_hat:

σ_hat = θ_hat[:σ]
