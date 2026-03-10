using Distributions

# Just for testing three groups, one with two observations 
# and one with three observations and one with one observation

# Helper function to make groups of equal size, last group may be smaller
function splitEqualGroups(y, nPerGroup)
    nElements = length(y)
    nGroups = ceil(Int, nElements/nPerGroup)
    Y = []
    i = 1
    while i <= nElements
        push!(Y, y[i:min(i+nPerGroup-1, nElements)])
        i += nPerGroup
    end
    return Y
end

# Data structure
y = [2, 3, 1, 0, 5, 4, 5] # All observations in a vecto
Y = splitEqualGroups(y, 1)

# ### Set up Poisson model
mutable struct PoissonRegParams 
    a::Float64
    μ::Float64
    σᵥ::Float64
    σ₀::Float64
    Z::Vector{Matrix{Float64}} # Covariates for each group
end

a = 0.8             # Persistence
μ = 1               # Unconditional log intensity  
σᵥ = 0.3            # State std deviation
σ₀ = 10             # Initial observation std deviation
Z = [randn(2,3), randn(2,3), randn(2,3), randn(1,3)] 

θ = PoissonRegParams(a, μ, σᵥ, σ₀, Z); # Instantiate the parameter struct for PGAS

function observation(θ, state, t) # time t is group time
    λs = exp.(θ.Z[t]*state) # θ.Z[t] This is a n_t × p matrix
    return product_distribution(Poisson.(λs)) # No broadcast, this is a multivariate dist
end

# Multi-dim state and multi-dim obs, also with singletons
state = randn(3)
p = length(state)
Z = [randn(length(Y[t]), p) for t in 1:length(Y)] # Just intercept 
for t in 1:length(Y)
    logp = logpdf(observation(θ, state, t), Y[t])
    println("t = $t - Number of observations is $(length(Y[t])) and logpdf is $logp")
end

# Univariate state and multi-dim obs, also with singletons
state = randn(1)
p = length(state)
Z = [randn(length(Y[t]),p) for t in 1:length(Y)] # Just intercept
θ = PoissonRegParams(a, μ, σᵥ, σ₀, Z); # Instantiate the parameter struct for PGAS
for t in 1:length(Y)
    logp = logpdf(observation(θ, state, t), Y[t])
    println("t = $t - Number of observations is $(length(Y[t])) and logpdf is $logp")
end

# multi-dim state and univariate obs, also with singletons
y = [2, 3, 1, 0, 5, 4, 5] # All observations in a vector
Y = splitEqualGroups(y, 1) # Each group has one observation
state = randn(3)
p = length(state)
Z = [randn(length(Y[t]),p) for t in 1:length(Y)] # Just intercept
θ = PoissonRegParams(a, μ, σᵥ, σ₀, Z); # Instantiate the parameter struct for PGAS
for t in 1:length(Y)
    logp = logpdf(observation(θ, state, t), Y[t])
    println("t = $t - Number of observations is $(length(Y[t])) and logpdf is $logp")
end

# univariate state and univariate obs, also with singletons
y = [2, 3, 1, 0, 5, 4, 5] # All observations in a vector
Y = splitEqualGroups(y, 1) # Each group has one observation
state = randn(1)
p = length(state)
Z = [randn(length(Y[t]),p) for t in 1:length(Y)] # Just intercept
θ = PoissonRegParams(a, μ, σᵥ, σ₀, Z); # Instantiate the parameter struct for PGAS
for t in 1:length(Y)
    logp = logpdf(observation(θ, state, t), Y[t])
    println("t = $t - Number of observations is $(length(Y[t])) and logpdf is $logp")
end

# Alternative version with scalar observations
function observation(θ, state, t) # time t is group time
    λ = exp(θ.Z[t]*state) # θ.Z[t] This is a scalar
    return Poisson(λ) # No broadcast, this is a multivariate dist
end

y = [2, 3, 1, 0, 5, 4, 5] # All observations in a vector
Y = splitEqualGroups(y, 1) # Each group has one observation
state = randn(1)
p = length(state)
Z = [randn(length(Y[t]),p) for t in 1:length(Y)] # Just intercept
θ = PoissonRegParams(a, μ, σᵥ, σ₀, Z); # Instantiate the parameter struct for PGAS
for t in 1:length(Y)
    logp = logpdf(observation(θ, state, t), Y[t])
    println("t = $t - Number of observations is $(length(Y[t])) and logpdf is $logp")
end





# Poisson with no covariates - version 1
observation(θ, state, t) = Poisson(exp(state[1]))

# same model for all obs in the group - will work with broadcasting 
@btime sum(logpdf.(observation(θ, state, t), Y[t]))

# Poisson with no covariates - version 2
g = length.(Y)
function observationAlt(θ, state, t)
    return product_distribution(fill(Poisson(exp(state[1])), g[t]))
end
@btime logpdf(observationAlt(θ, state, t), Y[t])



dist = MvNormal(zeros(5), 1.0) # Initial distribution for state
pdf.(dist, Ref(zeros(5))) # Evaluate pdf at zero vector



f(x) = 0.5*x + 1.0
x = 4
f(x)
 