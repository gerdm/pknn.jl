import Pknn

begin
    n_train = 101
    X, y = Pknn.Utils.make_moons(n_samples=250, noise=0.3, random_state=314)
    X_test = X[n_train + 1:end, :]
    y_test = y[n_train + 1:end]
    X = X[begin:n_train, :]
    y = y[begin:n_train]
end

# Initial sampling configuration
config = [
    Dict("beta"=>10, "k"=>1, "eta"=>1.1),
    Dict("beta"=>5, "k"=>5, "eta"=>1.1), 
]
target_samples = 3000
pknn = Pknn.PKC(X, y, config, target_samples)

burnout = 500
println("Sampling $target_samples rvs.")
@time begin
    pacc = Pknn.fit!(pknn, burnout)
end
println(pacc)

kch1 = Pknn.obtain_k_samples(pknn, 1)
kch2 = Pknn.obtain_k_samples(pknn, 2)
beta_ch1 = Pknn.obtain_beta_samples(pknn, 1)
beta_ch2 = Pknn.obtain_beta_samples(pknn, 2)
