""" 
    PGASsimulate!(X, y, p, N, param, prior, transition, observation, 
        initproposal, resampler, Xref = nothing)  


Single update step of the PGAS algorithm with `N` particles in X (N×p×T) to simulate from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):
- `prior` is the prior for the initial state p(x₀) 
- `transition` is the transition density p(xₜ | xₜ₋₁)
- `observation` is the observation density p(yₜ | xₜ)
param is a struct with all the parameters in the model needed to evaluate the prior, transition and observation densities.

`initproposal` is the proposal distribution for x₀ and `resampler` is the resampling function which defaults to systematic resampling.

Xref is a T×p matrix with conditioning particle path - if nothing, unconditional PF is run.
If sample_t0 is true, then sample also a t=0.
""" 
function PGASsimulate!(X, y, p, N, param, prior, transition, observation,  
    initproposal, resampler, Xref = nothing; sample_t0 = true) 

    conditioning = !isnothing(Xref)
    T = length(y)
    if sample_t0
        T = T + 1 # t = 1 now really corresponds to t=0 in y
    end
    a = zeros(Int, N, T)    # Ancestor indices
    w = zeros(N, T)         # Weights
    γ = zeros(N)
    logweights = zeros(N)
    ESSthreshold = 0.5*N
    ResampleCount = 0
    for t  = 1:T
        if t == 1
            for n in 1:N
                X[n,:,t] .= rand(initproposal)
            end  
            if conditioning
                X[N,:,t] = Xref[t,:]; # Last particle set to the reference particle
            end

            # Compute importance weights
            for n in 1:N
                x = @view X[n,:,t] # Particle state at time t
                if sample_t0
                    logweights[n] = logpdf(prior, (p==1) ? x[1] : x) - 
                        logpdf(initproposal, (p==1) ? x[1] : x)
                else
                    logweights[n] = logpdf(observation(param, (p==1) ? x[1] : x, t), y[t])+ 
                        logpdf(prior, (p==1) ? x[1] : x) - 
                        logpdf(initproposal, (p==1) ? x[1] : x)
                end
            end

            weights = exp.(logweights .- maximum(logweights))
            w[:,t] = weights/sum(weights) # Save the normalized weights

            ind = collect(1:N)
        else # t ≥ 2

            # Resampling
            resample = (ESS(w[:,t-1]) <= ESSthreshold)
            if resample
                ResampleCount += 1
                ind .= resampler(w[:,t-1])
            else
                ind .= 1:N # no resampling
            end
             
            # Ancestor sampling
            if conditioning
                for n = 1:N 
                    x = @view X[n,:,t-1]
                    xref = @view Xref[t,:]
                    γ[n] = logpdf(transition(param, (p==1) ? x[1] : x, t-sample_t0), 
                        (p==1) ? xref[1] : xref)
                end
                w_as = w[:,t-1] .* exp.(γ .- maximum(γ))
                w_as = w_as/sum(w_as)
                ind[N] = findfirst(rand(1) .<= cumsum(w_as))
            end
            if resample
                w[:,t-1] .= 1/N
            end

            # Store the ancestor indices
            a[:,t] = ind;

            # Propagate particles - bootstrap proposal
            for n in 1:N 
                x = @view X[ind[n],:,t-1] 
                X[n,:,t] .= rand(transition(param, (p==1) ? x[1] : x, t-sample_t0))
            end 
            if conditioning
                @views X[N,:,t] = Xref[t,:]; # Set the N:th particle to the conditioning
            end

            # Compute importance weights
            for n in 1:N
                x = @view X[n,:,t]
                logweights[n] = logpdf(observation(param, (p==1) ? x[1] : x, t-sample_t0), 
                    y[t-sample_t0]) 
            end

            weights = w[:,t-1] .* exp.(logweights .- maximum(logweights))

            w[:,t] = weights/sum(weights) # Save the normalized weights
        end
        
    end

    #println("Fraction resampling steps: $(ResampleCount/(T-1))")

    # Generate the trajectories from ancestor indices
    ind = a[:,T];
    for t = (T-1):-1:1
        X[:,:,t] = X[ind,:,t]
        ind = a[ind,t]
    end
    # Finally, sample a trajectory and return it
    J = findfirst(rand(1) .<= cumsum(w[:,T]))   
    return X[J,:,:]'

end




""" 
    PGASsampler(y, param, nDraws, N, prior, transition, observation, 
        initproposal = prior, resampler = systematic)

Uses the PGAS algorithm with `N` particles to simulate `nDraws` from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):
- `prior` is the prior for the initial state p(x₁) 
- `transition` is the transition density p(xₜ | xₜ₋₁)
- `observation` is the observation density p(yₜ | xₜ)

param is a struct with all the parameters and data (e.g. covariates) in the model needed to evaluate the prior, transition and observation densities.

`initproposal` is the proposal distribution for x₁ and `resampler` is the resampling function which defaults to systematic resampling. If sample_t0 is true, then sample also a t=0.
""" 
function PGASsampler(y, param, nDraws, N, prior, transition, observation, 
    initproposal = prior, resampler = systematic; sample_t0 = true)

    initproposal = initproposal(param)
    prior = prior(param)
    p = length(initproposal)
    T = length(y)
    Xdraws = zeros(sample_t0 + T, p, nDraws)
    X = zeros(N, p, T + sample_t0)

    # Initialize the state by running a PF
    Xdraw = PGASsimulate!(X, y, p, N, param, prior, transition, observation, 
        initproposal, resampler; sample_t0 = sample_t0)
        
    Xdraws[:,:,1] = Xdraw

    # Run MCMC loop
    for k = 2:nDraws 
        # Sample the states using PGAS
        Xdraw = PGASsimulate!(X, y, p, N, param, prior, transition, observation, 
            initproposal, resampler, Xdraw; sample_t0 = sample_t0)    
        Xdraws[:,:,k] = Xdraw      
    end

    return Xdraws#permutedims(Xdraws, [2,1,3]) # returns T×p×nDraws array

end 

