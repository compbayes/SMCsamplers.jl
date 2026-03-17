# # Time-varying AR model

# In this example we analyze the AR(p) model with p lags
#
# ```math
# \begin{equation*}
#   \phi_t(L)y_t = \epsilon_t , \quad \epsilon_t \sim N(0,\sigma_\varepsilon^2) 
# \end{equation*}
# ```
# where $\phi_t(L) = 1 - \phi_{1t} L - \phi_{2t} L^2 - \ldots - \phi_{pt} L^p$ is the regular AR polynomial.
#
# Define the state vector as $\boldsymbol{x}_t = (\phi_{1t},\ldots,\phi_{pt})^\top$. The  AR model can be written as a linear regression with Gaussian noise:
# ```math
#   y_t = \boldsymbol{z}_t^\top \boldsymbol{x}_t + \epsilon_t, \quad \epsilon_t \sim N(0,\sigma_\varepsilon^2)
# ```
#

# Assuming a simple Gaussian random walk evolution for the parameters, the time-varying SAR model can be written as a state space model:
# ```math
# \begin{align*}
#   y_t &= \boldsymbol{z}_t^\top \boldsymbol{x}_t + \epsilon_t, \quad \epsilon_t \sim N(0,\sigma_\varepsilon^2) \\
#   \boldsymbol{x}_t &= \boldsymbol{x}_{t-1} + \boldsymbol{\nu}_t, \quad \boldsymbol{\nu}_t \sim N(0,\sigma^2_v \boldsymbol{I}) \\
#   \boldsymbol{x}_0 &\sim N(0, \sigma^2_0 \boldsymbol{I})  
# \end{align*}
# ```

# We may restrict the parameters of the model so that the process is stable at all time periods, for example by using the transformations 
# $$\phi_t = \frac{\tilde\phi_t}{\sqrt{1 + \tilde\phi_t^2}}$$  
# to ensure that $|\phi_{t}| < 1$.
# This restriction brings a nonlinearity into the measurement model.

# ### First some preliminaries:
println(Base.active_project())
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, ForwardDiff, PDMats
using LinearAlgebra, Measures
using Utils: quantile_multidim
using Utils: mvcolors as colors

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Simulate data from a AR(1) model
p = 1
nState = p             
T = 500
T₊ = 1 + T # include presample values
ϕ(t) = 0.8*sin(2π*t/T₊) # Time-varying AR coefficient
σₑ = 0.2
y = zeros(T₊)
paramEvol = zeros(T₊)
for t = 2:T₊
    y[t] = ϕ(t)*y[t-1] + σₑ*randn() 
    paramEvol[t,:] .= ϕ(t)
end
timevect = 0:T

# ### Set up the AR model as a nonlinear regression
lag1 = [NaN;y[1:end-1]]     # Lag 1
Z = lag1
Z = Z[2:end, :]             # Remove first s rows with NaNs
y = y[2:end];               # Remove lost observations

 
# ### Plot the data and the time-varying parameters
p1 = plot(timevect, paramEvol, label = L"\phi_t", lw = 2, c = colors[2], 
    title ="parameter evolution", ylims = (-1,1), legend = :bottomleft)
p2 = plot(timevect, [NaN;y], label = "data", lw = 2, c = colors[1], xlabel = "time", 
    ylabel = "", legend = false, title = "time series")
plot(p1, p2, layout = (1,2), size = (800, 300), bottommargin = 5mm)


stable = false  # true parameters are restricted to stable region.
if stable
    restr(x) = x/sqrt(1 + x^2) 
    invrestr(y) = y/sqrt(1 - y^2)
else
    restr(x) = x
    invrestr(y) = y
end

# ### Set up AR model structure for PGAS and set static parameter values
mutable struct ARParams 
    σₑ::Float64
    σᵥ::Float64
    σ₀::Float64
    Z::Matrix{Float64}
end

prior(θ) = Normal(0, θ.σ₀)
transition(θ, state, t) = Normal(state, θ.σᵥ)  
function observation(θ, state, t)
    state = restr.(state) # Apply the restriction to the state
    return Normal(θ.Z[t,:]' ⋅ state, θ.σₑ)
end

σₑ = σₑ                 # Noise std deviation from static model
σᵥ = 0.1                # State std deviation
σ₀ = 1                  # Initial state std deviation
θ = ARParams(σₑ, σᵥ, σ₀, Z);

nSim = 1000;            # Number of samples from posterior

# ### PGAS sampling
nParticles = 100         # Number of particles for PGAS
sample_t0 = true         # Sample state at t=0 ?
PGASdraws = PGASsampler(y, θ, nSim, nParticles, prior, transition, 
    observation);
PGASdraws = restr.(PGASdraws) # Apply the restriction to the draws
PGASmedian = median(PGASdraws, dims = 3)[:,:,1];
PGASquantiles = quantile_multidim(PGASdraws, [0.025, 0.975], dims = 3);
    
plt = [];
titles = [L"\phi_{t}"];
legendPos = [:bottomleft];
for j = 1:nState
    plt_tmp = plot(timevect, paramEvol[:,j], lw = 2, 
        c = :black, linestyle = :solid, 
        label = "true", title = titles[j], legend = legendPos[j])
    plot!(plt_tmp, timevect, PGASmedian[:,j], fillrange = PGASquantiles[:,j,1],
        fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
    plot!(timevect, PGASmedian[:,j], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
    plot!(timevect, PGASmedian[:,j], lw = 1, c = :gray, linestyle = :solid, 
        label = "PGAS(N=$nParticles)")
    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,2), size = (800, 300), ylims = (-1.7,1.7), xlabel = "time", 
    bottommargin = 5mm)

# ### Set up the state space model for FFBS sampling with EKF, UKF etc
A = PDMat(I(nState))
B = zeros(nState)
U = zeros(T,1)
Y = y
Σₑ = θ.σₑ^2
Σₙ = θ.σᵥ^2 * I(nState)
μ₀ = zeros(nState)
Σ₀ = θ.σ₀^2 * I(nState)
function C(state, z)
    state = restr.(state)
    return z ⋅ state
end
Cargs = [Z[t,:] for t in 1:T];
∂C(state, z) = ForwardDiff.gradient(state -> C(state, z), state)';

# ### FFBS posterior sampling using the Extended Kalman filter (EKF)
EKFdraws = zeros(T + sample_t0, nState, nSim)
μ_filterEKF, Σ_filterEKF  = FFBSx!(EKFdraws, U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, 
    μ₀, Σ₀; filter_output = true);
EKFdraws = restr.(EKFdraws) # Apply the restriction to the draws
EKFmedian = median(EKFdraws, dims = 3)[:,:,1];
EKFquantiles = quantile_multidim(EKFdraws, [0.025, 0.975], dims = 3);
for j = 1:nState
    plot!(plt[j], timevect, EKFmedian[:,j], lw = 1, c = colors[3], linestyle = :solid, 
        label = "EKF(1)")
    plot!(plt[j], timevect, EKFquantiles[:,j,1], lw = 1, c = colors[3], 
        linestyle = :solid, label = nothing)
    plot!(plt[j], timevect, EKFquantiles[:,j,2], lw = 1, c = colors[3], 
        linestyle = :solid, label = nothing)
end
plot(plt..., layout = (1,2), size = (800, 300), ylims = (-1.7,1.7), xlabel = "time", 
    bottommargin = 5mm)


# ### FFBS posterior sampling using the Unscented Kalman filter (UKF)
α = 1; β = 0; κ = 0;
UKFdraws = zeros(T + sample_t0, nState, nSim)
FFBS_unscented!(UKFdraws, U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀; 
    α = α, β = β, κ = κ);
UKFdraws = restr.(UKFdraws) # Apply the restriction to the draws
UKFmedian = median(UKFdraws, dims = 3)[:,:,1]
UKFquantiles = quantile_multidim(UKFdraws, [0.025, 0.975], dims = 3);
for j = 1:nState
    plot!(plt[j], timevect, UKFmedian[:,j], lw = 1, c = colors[2], linestyle = :solid, 
        label = "UKF(1)")
    plot!(plt[j], timevect, UKFquantiles[:,j,1], lw = 1, c = colors[2], 
        linestyle = :solid, label = nothing)
    plot!(plt[j], timevect, UKFquantiles[:,j,2], lw = 1, c = colors[2], 
        linestyle = :solid, label = nothing)
end
plot(plt..., layout = (1,2), size = (800, 350),  ylims = (-1.7,1.7), xlabel = "time", 
    bottommargin = 5mm)


# ### FFBS posterior sampling using the Iterated Extended Kalman filter (IEKF)
plotIEKF = true
if plotIEKF
    maxIter = 10
    tol = 10^-4 # tolerance for convergence
    IEKFdraws = zeros(T + sample_t0, nState, nSim)
    μ_filterIEKF, Σ_filterIEKF = FFBSx!(IEKFdraws, U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, 
        μ₀, Σ₀, maxIter, tol; filter_output = true);
    IEKFdraws = restr.(IEKFdraws) # Apply the restriction to the draws
    IEKFmedian = median(IEKFdraws, dims = 3)[:,:,1];
    IEKFquantiles = quantile_multidim(IEKFdraws, [0.025, 0.975], dims = 3);
    for j = 1:nState
        plot!(plt[j], timevect, IEKFmedian[:,j], lw = 1, c = colors[4], linestyle = :solid, 
            label = "IEKF($maxIter)")
        plot!(plt[j], timevect, IEKFquantiles[:,j,1], lw = 1, c = colors[4], 
            linestyle = :solid, label = nothing)
        plot!(plt[j], timevect, IEKFquantiles[:,j,2], lw = 1, c = colors[4], 
            linestyle = :solid, label = nothing)
    end
    plot(plt..., layout = (1,2), size = (800, 300), ylims = (-1.7,1.7), xlabel = "time", 
        bottommargin = 5mm)
end

# ### FFBS posterior sampling using Iterated Extended Kalman filter + line search (IEKF-L)
plotIEKFL = true
if plotIEKFL
    linesearch = true
    maxIter = 10
    tol = 10^-4 # tolerance for convergence
    IEKFLdraws = zeros(T + sample_t0, nState, nSim)
    μ_filterIEKFL, Σ_filterIEKFL = FFBSx!(IEKFLdraws, U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, 
        μ₀, Σ₀, maxIter, tol, linesearch; filter_output = true);
    IEKFLdraws = restr.(IEKFLdraws) # Apply the restriction to the draws
    IEKFLmedian = median(IEKFLdraws, dims = 3)[:,:,1];
    IEKFLquantiles = quantile_multidim(IEKFLdraws, [0.025, 0.975], dims = 3);
    for j = 1:nState
        plot!(plt[j], timevect, IEKFLmedian[:,j], lw = 1, c = colors[5], 
            linestyle = :solid, label = "IEKF-L($maxIter)")
        plot!(plt[j], timevect, IEKFLquantiles[:,j,1], lw = 1, c = colors[5], 
            linestyle = :solid, label = nothing)
        plot!(plt[j], timevect, IEKFLquantiles[:,j,2], lw = 1, c = colors[5], 
            linestyle = :solid, label = nothing)
    end
    plot(plt..., layout = (1,2), size = (1400, 600), ylims = (-1.7,1.7), xlabel = "time", 
        bottommargin = 5mm)
end

# ### Laplace approximation with FFBS
plotLaplace = true
if plotLaplace
    maxIter = 10
    tol = 10^-4 # tolerance for convergence
    nFailure = Ref(0) # Persistent counter that gets updated when failures occur
    LaplaceDraws = zeros(T + sample_t0, nState, nSim)
    μ_filterLaplace, Σ_filterLaplace = FFBS_laplace!(LaplaceDraws, U, Y, A, B, Σₙ, 
        μ₀, Σ₀, observation, θ; filter_output = true, nFailure = nFailure);
    LaplaceDraws = restr.(LaplaceDraws) # Apply the restriction to the draws
    Laplacemedian = median(LaplaceDraws, dims = 3)[:,:,1];
    Laplacequantiles = quantile_multidim(LaplaceDraws, [0.025, 0.975], dims = 3);
    for j = 1:nState
        plot!(plt[j], timevect, Laplacemedian[:,j], lw = 1, c = colors[6], 
            linestyle = :solid, label = "Laplace")
        plot!(plt[j], timevect, Laplacequantiles[:,j,1], lw = 1, c = colors[6], 
            linestyle = :solid, label = nothing)
        plot!(plt[j], timevect, Laplacequantiles[:,j,2], lw = 1, c = colors[6], 
            linestyle = :solid, label = nothing)
    end
    plot(plt..., layout = (1,2), size = (1400, 600), ylims = (-1.7,1.7), xlabel = "time", 
        bottommargin = 5mm)
end
println("Laplace failed at $(100*nFailure[]/nSim)% of the simulated trajectories") 