import Pknn
using Plots
using StatsPlots

begin
    X, y = Pknn.Utils.make_moons(n_samples=200, noise=0.3, random_state=314)
    X_test = X[151:end, :]
    y_test = y[151:end]
    X = X[begin:150, :]
    y = y[begin:150]
end

# Initial sampling configuration
config = [
    Dict("beta"=>10, "k"=>1, "eta"=>1.1),
    Dict("beta"=>5, "k"=>5, "eta"=>1.1), 
]
target_samples = 1500
pknn = Pknn.PKC(X, y, config, target_samples)

burnout = 500
println("Start sampling")
pacc = Pknn.fit!(pknn, burnout)
println(pacc)

kch1 = Pknn.obtain_k_samples(pknn, 1)
kch2 = Pknn.obtain_k_samples(pknn, 2)
beta_ch1 = Pknn.obtain_beta_samples(pknn, 1)
beta_ch2 = Pknn.obtain_beta_samples(pknn, 2)

begin
    l = @layout [a b; c [d; e]]
    colors = [yn == 1 ? "darkorange" : "turquoise3" for yn in y]
    Nx, Ny = 50, 50
    p1 = Pknn.plot_surface(pknn, Nx, Ny)
    scatter!(X[:, 1], X[:, 2], color=colors,
                label=nothing, markerstrokewidth=0, marker=:+)
    plot!(title="Dataset")

    p2 = Pknn.Utils.freqplot(kch1, label="k s(1)", alpha=0.5)
    Pknn.Utils.freqplot!(kch1, label="k s(2)", alpha=0.5)
    plot!(title="k-samples")

    p3 = histogram(beta_ch1, label="beta s(1)", alpha=0.5)
    histogram!(beta_ch2, label="beta s(2)", alpha=0.5)
    plot!(title="β-samples")

    p4 = plot(beta_ch1, label=nothing)
    plot!(beta_ch2, label=nothing, title="β-hist")

    p5 = plot(kch1, label=nothing)
    plot!(kch2, label=nothing)
    plot!(title="k-hist")

    plot(p1, p2, p3, p4, p5, layout=l)
end
