module pknn

using Random
using Einsum
using Statistics

# Use to create new dimensions
new = [CartesianIndex()]

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

"""
Compute the squared L2-norm between an input vector x
and target vector X

"""
function l2_distance(x, X)
    @einsum D[n1, n2] := (x[n1, m] - X[n2, m]) ^ 2
    return D
end


function find_k_closest(X; k=3)
    D = l2_distance(X, X)
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

end # module
