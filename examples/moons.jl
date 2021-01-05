import Pknn

using Plots
using StatsPlots

X, y = Pknn.Utils.make_moons(n_samples=150, noise=0.3, random_state=314)
target_samples = 3_000
config = [
    Dict("beta"=>10, "k"=>1, "eta"=>1.1),
    Dict("beta"=>5, "k"=>5, "eta"=>1.1), 
]
println("start sampling")
samples, pacc = Pknn.Model.sample_knn(X, y, config; target_samples=target_samples)
burnout = 1500
samples = samples[:, burnout:end, :]

k_samples = samples[:, :, 1]
beta_samples = samples[:, :, 2]
N_hist = 1:size(k_samples)[2]

active_samples = hcat(reshape(k_samples, (:,1)), reshape(beta_samples, (:, 1)))

l = @layout [a b; c [d; e]]
colors = [yn == 1 ? "darkorange" : "turquoise3" for yn in y]
Nx, Ny = 50, 50
p1 = Pknn.Utils.plot_surface(X, y, active_samples, Nx, Ny)
scatter!(X[:, 1], X[:, 2], color=colors,
             label=nothing, markerstrokewidth=0, marker=:+)
plot!(title="Dataset")

p2 = Pknn.Utils.freqplot(k_samples[1, :], label="k s(1)", alpha=0.5)
Pknn.Utils.freqplot!(k_samples[2, :], label="k s(2)", alpha=0.5)
plot!(title="k-samples")

p3 = histogram(beta_samples[1, :], label="beta s(1)", alpha=0.5)
histogram!(beta_samples[2, :], label="beta s(2)", alpha=0.5)
plot!(title="β-samples")

p4 = plot(beta_samples[1, :], label=nothing)
plot!(beta_samples[2, :], label=nothing, title="β-hist")

p5 = plot(k_samples[1, :], label=nothing)
plot!(k_samples[2, :], label=nothing)
plot!(title="k-hist")

plot(p1, p2, p3, p4, p5, layout=l)