using Pkg
dir = "/Users/gerardoduran/Documents/repos/pknn"
Pkg.activate(dir)
# The file
import pknn
using Plots
using StatsPlots

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

X, y = pknn.make_moons(n_samples=150, noise=0.3, random_state=314)

target_samples = 3_000
config = [
    Dict("beta"=>10, "k"=>1, "eta"=>1.1),
    Dict("beta"=>5, "k"=>5, "eta"=>1.1), 
]
samples, pacc = pknn.sample_knn(X, y, config; target_samples=target_samples)
burnout = 1500
samples = samples[:, burnout:end, :]

colors = [yn == 1 ? "crimson" : "deepskyblue3" for yn in y]
k_samples = samples[:, :, 1]
beta_samples = samples[:, :, 2]
N_hist = 1:size(k_samples)[2]

l = @layout [a b; c [d; e]]
p1 = scatter(X[:, 1], X[:, 2], color=colors, label=nothing, title="Dataset")

p2 = freqplot(k_samples[1, :], label="k s(1)", alpha=0.5)
freqplot!(k_samples[2, :], label="k s(2)", alpha=0.5)

p3 = histogram(beta_samples[1, :], label="beta s(1)", alpha=0.5)
histogram!(beta_samples[2, :], label="beta s(2)", alpha=0.5)

p4 = plot(beta_samples[1, :], label=nothing)
plot!(beta_samples[2, :], label=nothing, title="β-hist")

p5 = plot(k_samples[1, :], label=nothing)
plot!(k_samples[2, :], label=nothing)
plot!(title="k-hist")

plot(p1, p2, p3, p4, p5, layout=l, dpi=200)
