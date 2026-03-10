""" 
    PGASupdate(y, p, N, θ, prior, transition, observation, 
        initproposal, resampler, Xref = nothing)  


Single update step of the PGAS algorithm with `N` particles to simulate from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):
- `prior` is the prior for the initial state p(x₁) 
- `transition` is the transition density p(xₜ | xₜ₋₁)
- `observation` is the observation density p(yₜ | xₜ)
θ is a struct with all the parameters in the model needed to evaluate the prior, transition and observation densities.

`initproposal` is the proposal distribution for x₁ and `resampler` is the resampling function which defaults to systematic resampling.

Xref is a p×T matrix with conditioning particle path - if nothing, unconditional PF is run
""" 
function PGASupdate(y, p, N, θ, prior, transition, observation,  
    initproposal, resampler, Xref = nothing; sample_t0 = true) 

    conditioning = !isnothing(Xref)
    T = length(y)
    if sample_t0
        T = T + 1
    end
    X = zeros(N, p, T)      # Particles 
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
                X[N,:,t] = Xref[:,t] # Last particle conditions on reference
            end
            # Compute importance weights
            for n in 1:N               
                if sample_t0
                    logweights[n] = logpdf(prior, X[n,:,t]) - 
                        logpdf(initproposal, X[n,:,t])
                else
                    logweights[n] = logpdf(observation(θ, X[n,:,t], t), y[t])  + 
                        logpdf(prior, X[n,:,t]) - logpdf(initproposal, X[n,:,t])
                end
            end

            weights = exp.(logweights .- maximum(logweights))
            w[:,t] = weights/sum(weights) # Save the normalized weights
            ind = 1:N
        else # t ≥ 2
            resample = (ESS(w[:,t-1]) <= ESSthreshold)
            if resample
                ResampleCount += 1
                ind = resampler(w[:,t-1])
            else
                ind = 1:N # no resampling
            end

            for n in 1:N
                X[n,:,t] .= rand(transition(θ, X[ind[n],:,t-1], t-sample_t0))
            end 
             
            if conditioning 
                X[N,:,t] = Xref[:, t] # N:th particle to the conditioning
            end
            if conditioning && resample
                # Ancestor sampling
                for n = 1:N 
                    γ[n] = logpdf(transition(θ, X[n,:,t-1], t-sample_t0), Xref[:,t])
                end
                w_as = w[:,t-1] .* exp.(γ .- maximum(γ))
                w_as = w_as/sum(w_as)
                ind[N] = findfirst(rand(1) .<= cumsum(w_as))
                w[:,t-1] .= 1/N
            end

            # Store the ancestor indices
            a[:,t] = ind;

            # Compute importance weights
            for n in 1:N
                logweights[n] = logpdf(observation(θ, X[n,:,t], t-sample_t0), 
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
        X[:,:, t] = X[ind,:,t]
        ind = a[ind,t]
    end
    # Finally, sample a trajectory and return it
    J = findfirst(rand(1) .<= cumsum(w[:,T]))   
    return X[J,:,:] # Maybe also return the particle system: X, w

end




""" 
    PGASsampler(y, θ, nDraws, N, prior, transition, observation, 
        initproposal = prior, resampler = systematic)

Uses the PGAS algorithm with `N` particles to simulate `nDraws` from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):
- `prior` is the prior for the initial state p(x₁) 
- `transition` is the transition density p(xₜ | xₜ₋₁)
- `observation` is the observation density p(yₜ | xₜ)

θ is a struct with all the parameters in the model needed to evaluate the prior, transition and observation densities.

`initproposal` is the proposal distribution for x₁ and `resampler` is the resampling function which defaults to systematic resampling. If sample_t0 is true, then sample also a t=0.
""" 
function PGASsampler(y, θ, nDraws, N, prior, transition, observation, 
    initproposal = prior, resampler = systematic; sample_t0 = true)

    initproposal = initproposal(θ)
    prior = prior(θ)
    p = length(initproposal)
    T = length(y)
    Xdraws = zeros(p, sample_t0 + T, nDraws)

    # Initialize the state by running a PF
    println("hej mor ❤️ ") #FIXME: remove this debug info
    X = PGASupdate(y, p, N, θ, prior, transition, observation, initproposal, resampler; 
        sample_t0 = sample_t0)
        
    Xdraws[:,:,1] = X

    # Run MCMC loop
    for k = 2:nDraws 
        println("PGAS iteration $k / $nDraws")
        # Sample the states using PGAS
        X = PGASupdate(y, p, N, θ, prior, transition, observation, initproposal,
            resampler, Xdraws[:,:,k-1]; sample_t0 = sample_t0)    
        Xdraws[:,:,k] = X      
    end

    return permutedims(Xdraws, [2,1,3]) # returns T×p×nDraws array

end 

