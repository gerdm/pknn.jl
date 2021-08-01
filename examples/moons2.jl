import Pknn
using Plots
using StatsPlots

function train_test_moons(n_samples, n_train, noise, random_state=nothing)
    X, y = Pknn.Utils.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_test = X[n_train+1:end, :]
    y_test = y[n_train+1:end]

    X = X[begin:n_train, :]
    y = y[begin:n_train]

    return (X, y), (X_test, y_test)
end

function sample_and_plot(n_samples, n_train, noise, burnout, config, target_samples, random_state=nothing)
    (X, y), (X_test, y_test) = train_test_moons(n_samples, n_train, noise, random_state)
    colors = [yn == 1 ? "darkorange" : "turquoise3" for yn in y]
    pknn = Pknn.PKC(X, y, config, target_samples)
    pacc = Pknn.fit!(pknn, burnout)
    kch1 = Pknn.obtain_k_samples(pknn, 1)
    kch2 = Pknn.obtain_k_samples(pknn, 2)

    Nx, Ny = 50, 50
    p1 = Pknn.plot_surface(pknn, Nx, Ny)
    scatter!(X[:, 1], X[:, 2], color=colors,
                label=nothing, markerstrokewidth=0, marker=:+)
    plot!(title="Pknn | samples=$n_train")
    p2 = Pknn.Utils.freqplot(kch1, label="k s(1)", alpha=0.5)
    Pknn.Utils.freqplot!(kch1, label="k s(2)", alpha=0.5)
    plot!(title="k-samples", ylabel="frequency", xlabel="number neighbours")
    xlims!(0, 20)
    plot(p1, p2, size=(900, 300))
end

begin
    random_state = 314
    n_samples, noise = 200, 0.3
    burnout, target_samples = 500, 150
    config = [
        Dict("beta"=>10, "k"=>1, "eta"=>1.1),
        Dict("beta"=>5, "k"=>5, "eta"=>1.1), 
    ]
end

n_train = 22
sample_and_plot(n_samples, n_train, noise, burnout, config, target_samples, random_state)

values = 22:2:200
n_its = length(values)
anim = @animate for (i, n_train) ∈ enumerate(values)
    println("@it $i/$n_train")
    sample_and_plot(n_samples, n_train, noise, burnout, config, target_samples, random_state)
end
gif(anim, "moons_evolve_freq.gif", fps=15)
