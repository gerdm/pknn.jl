module Model

using Random
using Einsum
using Statistics

# Use to create new dimensions
new = [CartesianIndex()]


"""
Compute the squared L2-norm between an input vector x
and target vector X
"""
function l2_distance(x, X)
    @einsum D[n1, n2] := (x[n1, m] - X[n2, m]) ^ 2
    return D
end


function find_k_closest(X, Z=nothing; k=3)
    if Z === nothing
        Z = X
    end
    D = l2_distance(Z, X)
    # k-closest ommiting the element itself
    k_closest = mapslices(sortperm, D; dims=2)[:, 2:k+1]
    return k_closest
end


function compute_likelihood(X, y; beta, k)
    K = length(unique(y))
    range_k = 0:K-1
    k_closest = find_k_closest(X, k=k)

    num = y[k_closest] .== y
    num = exp.(beta * mean(num, dims=2))
    den = y[k_closest] .== range_k[new, new, :]
    den = sum(exp.(beta * mean(den, dims=2)), dims=3)

    # Remove singleton dimensions
    num = dropdims(num, dims=2)
    den = dropdims(den, dims=(2, 3))

    likelihood = prod(num ./ den)
    return likelihood
end


function knn_mcmc(X, y; k, beta, target_samples=10_000, eta=1.,
                  print_progress=false)
    samples = zeros(target_samples, 2)
    n_rounds, n_samples = 0, 0
    while n_samples < target_samples
        n_rounds += 1

        if (n_rounds % 100 == 0) & print_progress
            print("@it $n_rounds | samples=$n_samples\r")
        end

        beta_hat = abs(beta + randn() * eta)
        k_hat = abs(k + rand(-4:4))
        k_hat = max(1, k_hat)

        L_hat = compute_likelihood(X, y, beta=beta_hat, k=k_hat)
        L = compute_likelihood(X, y, beta=beta, k=k)
        A = min(1, L_hat / L)

        if A > rand()
            n_samples += 1
            samples[n_samples, 1] = k_hat
            samples[n_samples, 2] = beta_hat
            k, beta = k_hat, beta_hat
        end
    end

    return samples, target_samples / n_rounds
end

function sample_knn(X, y, configs; target_samples)
    n_config = length(configs)
    samples_v = zeros(n_config, target_samples, 2)
    pacc_v = zeros(n_config)

    Threads.@threads for i=1:n_config
        config = configs[i]
        beta = config["beta"]
        k = config["k"]
        eta = config["eta"]
        samples, pacc = knn_mcmc(X, y; k=k, beta=beta, eta=eta,
                 target_samples=target_samples)
        samples_v[i, :, :] = samples
        pacc_v[i] = pacc
    end
    return samples_v, pacc_v
end


function infer(yn, Xn, X, y, samples)
    k_samples = samples[:, 1]
    beta_samples = samples[:, 2]
    n_samples = length(k_samples)
    K = length(unique(y))
    range_k = 0:K-1

    n_test, _ = size(Xn)
    P_values = zeros(n_test, n_samples)
    D = l2_distance(Xn, X)
    closest = mapslices(sortperm, D; dims=2)

    Threads.@threads for i=1:n_samples
        ki = Int(k_samples[i])
        βi = beta_samples[i]
        k_closest = closest[:, 1:ki]

        num = y[k_closest] .== yn
        num = exp.(βi * mean(num, dims=2))
        den = y[k_closest] .== range_k[new, new, :]
        den = sum(exp.(βi * mean(den, dims=2)), dims=3)

        # Remove singleton dimensions
        num = dropdims(num, dims=2)
        den = dropdims(den, dims=(2, 3))
        proba = num ./ den

        P_values[:, i] = proba
    end

    return mean(P_values, dims=2)
end

end #module