module Utils

using Random
using Plots: plot, plot!, contourf, bar, bar!
import ..Model: infer

"""
Make two interleaving half circles.
(Taken from Python's scikit-learn)
"""
function make_moons(; n_samples::Int=100, shuffle_rows::Bool=true, noise::Float64=0.0,
                    random_state=nothing)
    if random_state != nothing
        Random.seed!(random_state)
    end
    n_samples_out = n_samples ÷ 2
    n_samples_in = n_samples - n_samples_out

    outer = range(0, stop=π, length=n_samples_out)
    inner = range(0, stop=π, length=n_samples_in)
    outer_circ_x = cos.(outer)
    outer_circ_y = sin.(outer)
    inner_circ_x = 1 .- cos.(inner)
    inner_circ_y = 1 .- sin.(inner) .- 0.5

    X = [outer_circ_x outer_circ_y; inner_circ_x inner_circ_y]
    y = [ones(n_samples_out); zeros(n_samples_in)]

    if noise >= 0
        X = X .+ randn(n_samples) * noise
    end

    if shuffle_rows
        new_indices = shuffle(1:n_samples)
        X = X[new_indices, :]
        y = y[new_indices]
    end

    return X, y
end

function frequency(x)
    elements = Dict{Float64, Int64}()
    for xi in x
        if xi in keys(elements)
            elements[xi] += 1
        else
            elements[xi] = 1
        end
    end

    elements = hcat(collect(keys(elements)), collect(values(elements)))
    elements = elements[sortperm(elements[:, 1]), :]
    return elements
end

function freqplot(x; kwargs...)
    elements = frequency(x)
    values, freq = elements[:, 1], elements[:, 2]
    print(kwargs)
    bar(values, freq; kwargs...)
end

function freqplot!(x; kwargs...)
    elements = frequency(x)
    values, freq = elements[:, 1], elements[:, 2]
    print(kwargs)
    bar!(values, freq; kwargs...)
end

function plot_surface(X, y, samples, Nx, Ny)
    xmin, ymin = minimum(X, dims=1)
    xmax, ymax = maximum(X, dims=1)
    xx = range(xmin, xmax, length=Nx)
    yy = range(ymin, ymax, length=Ny)
    D = cat(xx' .* ones(Nx), ones(Ny)' .* yy, dims=3)
    D = reshape(D, (:, 2))
    P_res = infer(1, D, X, y, samples)
    P_res = reshape(P_res, (Nx, Ny))
    contourf(xx, yy, P_res, c=:RdBu, linewidth=0)
    plot!(xlim=(xmin, xmax), ylim=(ymin, ymax))
end
    
end # Utils' module