# # Logistic regression

import MaximumLikelihoodProblems

import Distributions
import ForwardDiff
import LogDensityProblems
import StatsFuns
import TransformVariables

struct LogisticRegression{Ty, TX}
    y::Ty
    X::TX
end

function (problem::LogisticRegression)(θ)
    y = problem.y
    X = problem.X

    β = θ.β

    η = X * β
    μ = StatsFuns.logistic.(η)
    log_likelihood = sum(Distributions.logpdf.(Distributions.Bernoulli.(μ), y))
    return log_likelihood
end

N = 10_000

## the first column (the column of all ones) is the intercept
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

# get the standard errors:

θ_hat_stderror = MaximumLikelihoodProblems.stderror(transformed_gradient_problem,
                                                        θ_hat)
β_hat_stderror = θ_hat_stderror[:β]

# get 95% confidence intervals:

θ_hat_95_CI_lower, θ_hat_95_CI_upper = MaximumLikelihoodProblems.confint(transformed_gradient_problem,
                                                                                         θ_hat;
                                                                                         level = 0.95)

# β_hat:

β_hat

# lower bound of 95% confidence interval for β_hat:

β_hat_95_CI_lower = θ_hat_95_CI_lower[:β]

# upper bound of 95% confidence interval for β_hat:

β_hat_95_CI_upper = θ_hat_95_CI_upper[:β]
