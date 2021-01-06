module Pknn

include("Model.jl")
include("Utils.jl")

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
