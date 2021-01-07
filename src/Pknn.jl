module Pknn

include("Model.jl")
include("Utils.jl")

"""
    PKC(X, y, config, target_samples)

Struct to work with a probabilistic K-nearest neighbours model
with l2 distance

# Arguments
- `X::Array{Float64, 2}`: A features' matrix of asidsda
    M features and N observations
- `y::Array{Float64, 1}`: The target matrix of N binary (0,1) observations
- `config::Array{Dict{String, Real}, 1}`: An array of dictionaries containing
    the initial values for k,beta; as well as the step-size eta
- `target_samples::Int64`: The number of samples per initial configuration
"""
mutable struct PKC
    X::Array{Float64, 2}
    y::Array{Float64, 1}
    config::Array{Dict{String, Real}, 1}
    target_samples::Int64
    samples::Array{Float64, 3}

    function PKC(X, y, config, target_samples)
        n_chains = length(config)
        samples = zeros((n_chains, target_samples, 2))

        return new(X, y, config, target_samples, samples)
    end

end

"""
    fit!(instance, burnout)

Sample each configuration to obtain posterior samples for k and beta.
This function returns an array of average accepted values and assigns
posterior samples to the target_samples' matrix of the instance.
"""
function fit!(instance::PKC, burnout::Int64=1200)
    total_samples = burnout + instance.target_samples
    samples, pacc = Pknn.Model.sample_knn(instance.X, instance.y,
                    instance.config,
                    target_samples=total_samples)
    instance.samples = samples[:, burnout+1:end, :]

    return pacc
end

function plot_surface(instance::PKC, Nx, Ny)
    N, M = size(instance.X)
    if M != 2
        error("X is not 2-dimensional")
    end

    active_samples = reshape(instance.samples, (:, 2))
    Pknn.Utils.plot_surface(instance.X, instance.y,
                            active_samples, Nx, Ny)
end

function k_samples(instance::PKC, chain=nothing)
    samples = instance.samples[:, :, 1]
    if chain === nothing
        samples = reshape(samples, :)
    else
        samples = samples[chain, :]
    end
    
    return samples
end

function beta_samples(instance::PKC, chain=nothing)
    samples = instance.samples[:, :, 2]
    if chain === nothing
        samples = reshape(samples, :)
    else
        samples = samples[chain, :]
    end
    
    return samples
end

end # module
