# # Logistic regression

import MaximumLikelihoodProblems

import Distributions
import ForwardDiff
import LogDensityProblems
import Parameters
import StatsFuns
import TransformVariables

struct LogisticRegression{Ty, TX}
    y::Ty
    X::TX
end

function (problem::LogisticRegression)(θ)
    Parameters.@unpack y, X = problem
    Parameters.@unpack β = θ
    η = X*β
    μ = StatsFuns.logistic.(η)
    loglik = sum(Distributions.logpdf.(Distributions.Bernoulli.(μ), y))
    return loglik
end

N = 10_000
X = hcat(ones(N), randn(N))
size_β = (2,)
β_true = [1.0, 2.0]
η_true = X * β_true
μ_true = StatsFuns.logistic.(η_true)
y = rand.(Distributions.Bernoulli.(μ_true))

problem = LogisticRegression(y, X)
transformation = TransformVariables.as((β = TransformVariables.as(Array, size_β), ))
transformed_problem = LogDensityProblems.TransformedLogDensity(transformation,
                                                               problem)
transformed_gradient_problem = LogDensityProblems.ADgradient(:ForwardDiff,
                                                             transformed_problem)

β_hat_initial_guess = zeros(size_β)
θ_hat_initial = (; β = β_hat_initial_guess)

# θ_hat:

θ_hat = MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                      θ_hat_initial)

# β_hat:

β_hat = θ_hat[:β]
