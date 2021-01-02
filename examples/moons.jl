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


X, y = pknn.make_moons(n_samples=150, noise=0.3, random_state=314)

k0, beta0 = 1, 10
eta = 1.1
target_samples = 5_000
samples, pacc = pknn.knn_mcmc(X, y, k=k0, beta=beta0,
                target_samples=target_samples, eta=eta)

burnout = 1500
samples = samples[burnout:end, :]

colors = [yn == 1 ? "crimson" : "deepskyblue3" for yn in y]
k_samples = samples[:, 1]
beta_samples = samples[:, 2]
N_hist = 1:length(k_samples)
p1 = scatter(X[:, 1], X[:, 2], color=colors, label=nothing, title="Dataset")
p2 = freqplot(k_samples, label="k-samples")
p3 = histogram(beta_samples, label="beta-samples")
p4 = plot(beta_samples, label="beta-hist")
plot!(N_hist, N_hist .* NaN, label="k-hist", color="orange")
plot!(twinx(), k_samples, color="orange", label=nothing)
plot(p1, p2, p3, p4)