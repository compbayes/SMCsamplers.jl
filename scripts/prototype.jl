# To specify the different model in a unified interface:

# Linear homogenous
transition(θ, state, t) = θ.A
observation(θ, state, t) = θ.C

# Linear non-homogenous
transition(θ, state, t) = θ.A[:,:,t]
observation(θ, state, t) = θ.C[:,:,t]

# Nonlinear function
transition(θ, state, t) = θ.h(state)  
observation(θ, state, t) = θ.g(state)

