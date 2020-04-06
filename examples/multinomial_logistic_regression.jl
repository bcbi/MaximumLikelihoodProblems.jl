# # Multinomial logistic regression

import MaximumLikelihoodProblems

import Distributions
import ForwardDiff
import LogDensityProblems
import NNlib
import TransformVariables

struct MultinomialLogisticRegression{Ty, TX}
    y::Ty
    X::TX
end

function (problem::MultinomialLogisticRegression)(θ)
    y = problem.y
    X = problem.X

    β = θ.β

    num_rows = size(X, 1)
    num_covariates = size(β, 1) # `size(β, 1)` is equal to `size(X, 2)`
    num_classes = size(β, 2) + 1

    ## the first column of all zeros corresponds to the base class
    ## i.e. the coefficient β₀ for the base class is always fixed to be zero
    β_with_base_class = hcat(zeros(num_covariates), β)
    η = X * β_with_base_class

    μ = NNlib.softmax(η; dims=2)
    log_likelihood = sum([Distributions.logpdf(Distributions.Multinomial(1, μ[i, :]), y[i, :]) for i = 1:num_rows])
    return log_likelihood
end

N = 10_000

## the first column (the column of all ones) is the intercept
X = hcat(ones(N), randn(N))

size_β = (2, 3)
β_true = [1.0 2.0 3.0; 4.0 5.0 6.0]
num_covariates = size(β_true, 1)
β_true_with_base_class = hcat(zeros(num_covariates), β_true)
η_true = X * β_true_with_base_class
μ_true = NNlib.softmax(η_true; dims=2)
y = vcat([rand(Distributions.Multinomial(1, μ_true[i,:]))' for i in 1:N]...)

problem = MultinomialLogisticRegression(y, X)
transformation = TransformVariables.as((β = TransformVariables.as(Array, size_β), ))
transformed_problem = LogDensityProblems.TransformedLogDensity(transformation,
                                                               problem)
transformed_gradient_problem = LogDensityProblems.ADgradient(:ForwardDiff,
                                                             transformed_problem)

β_hat_initial_guess = zeros(size_β)
θ_hat_initial = (; β = β_hat_initial_guess)

# θ_hat:

θ_hat = MaximumLikelihoodProblems.fit(transformed_gradient_problem,
                                      θ_hat_initial;
                                      show_progress_meter = false)

# β_hat:

β_hat = θ_hat[:β]

# β\_hat\_with\_base\_class:

num_covariates = size(β_hat, 1)
β_hat_with_base_class = hcat(zeros(num_covariates), β_hat)
