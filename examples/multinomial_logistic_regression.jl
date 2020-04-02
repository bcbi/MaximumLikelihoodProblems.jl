# # Multinomial logistic regression

import MaximumLikelihoodProblems

import Distributions
import ForwardDiff
import LogDensityProblems
import Parameters
import NNlib
import TransformVariables

struct MultinomialLogisticRegression{Ty, TX}
    y::Ty
    X::TX
end

function (problem::MultinomialLogisticRegression)(θ)
    Parameters.@unpack y, X = problem
    Parameters.@unpack β = θ
    num_rows, num_covariates = size(X)
    num_classes = size(β, 2) + 1

    ## the first column of all zeros corresponds to the base class
    η = X * hcat(zeros(num_covariates), β)

    μ = NNlib.softmax(η; dims=2)
    loglik = sum([Distributions.logpdf(Distributions.Multinomial(1, μ[i, :]), y[i, :]) for i = 1:num_rows])
    return loglik
end

N = 10_000
X = hcat(ones(N), randn(N))
size_β = (2, 3)
β_true = [1.0 2.0 3.0; 4.0 5.0 6.0]
η_true = X * hcat(zeros(2), β_true)
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
                                      show_progress_bar = false)

# β_hat:

β_hat = θ_hat[:β]
