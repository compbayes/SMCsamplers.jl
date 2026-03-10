# # Stochastic volatility model

# In this example we explore the joint posterior of the state $x_t$ in a simple stochastic volatility (SV) model with known static parameters: 
#
# ```math
# \begin{align*}
#   y_t &= \exp(x_t/2)\epsilon_t,  \quad \epsilon_t \sim N(0,1) \\
#   x_t &= ax_{t-1} + \nu_t, \quad \nu_t \sim N(0,\sigma_v) \\
#   x_0 &\sim N(0, \sigma_0)  
# \end{align*}
# ```

# First some preliminaries:
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Set up SV model structure for PGAS and set static parameter values
mutable struct SVParams 
    a::Float64
    σᵥ::Float64
    σ₀::Float64
end
prior(θ) = Normal(0, θ.σ₀)
transition(θ, state, t) = Normal(θ.a * state, θ.σᵥ)  
observation(θ, state, t) = Normal(0, exp(state/2));   

a = 0.9         # Persistence
σᵥ = 1          # State std deviation
σ₀ = 0.5        # Initial observation std deviation
θ = SVParams(a, σᵥ, σ₀); # Set up parameter struct for PGAS

# ### Simulate data from SV model
T = 200         # Length of time series
x = zeros(T)
y = zeros(T)
x0 = rand(prior(θ))
for t in 1:T
    if t == 1
        x[t] = rand(transition(θ, x0, t))
    else
        x[t] = rand(transition(θ, x[t-1], t))
    end
    y[t] = rand(observation(θ, x[t], t))
end 
plt1 = plot(exp.(x/2); label="", xlabel="t", lw = 2, legend = :topleft, 
    color = colors[3], title = "Standard deviation")
plt2 = plot(y; seriestype=:scatter, label="", xlabel="t", markersize = 2,
    color = colors[1], markerstrokecolor = :auto, title = "Realized observations")
plot(plt1, plt2, layout = (2,1))

# ### PGAS sampling
Nₚ = 20         # Number of particles for PGAS
Nₛ = 1000       # Number of samples from posterior
PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation)
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = quantile_multidim(PGASdraws, [0.025, 0.975], dims = 3);
    
# ### Plot the posterior mean and 95% credible intervals from both algorithms
plt = plot(x, c = colors[3], lw = 1, label = "true state")
plot!(PGASmean[:,1], lw = 1, c = colors[1], linestyle = :solid, label = "PGAS(N=$Nₚ)")
plot!(PGASquantiles[:,1,1], fillrange = PGASquantiles[:,1,2],
        fillalpha = 0.2, fillcolor = colors[1], linecolor = colors[1], label = "", lw = 0) 