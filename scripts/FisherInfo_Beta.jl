#### Fisher Information Matrix for Beta Distribution

using LinearAlgebra
using SpecialFunctions   # for trigamma

function fisher_beta_blocks(Xm, Xp, βm, βp)

    T  = size(Xm,1)
    pm = size(Xm,2)
    pp = size(Xp,2)

    ηm = Xm * βm
    ηp = Xp * βp

    expηm = exp.(ηm)
    expηp = exp.(ηp)

    μ = 1.0 ./ (1.0 .+ exp.(-ηm))
    ϕ = expηp

    a = μ .* ϕ
    b = (1 .- μ) .* ϕ

    ψa = digamma.(a)
    ψb = digamma.(b)
    ψϕ = digamma.(ϕ)

    ψ1a = trigamma.(a)
    ψ1b = trigamma.(b)
    ψ1ϕ = trigamma.(ϕ)

    E1 = ψa .- ψϕ
    E2 = ψb .- ψϕ

    ∇Bₘ = (ψa .- ψb) .* ϕ
    ∇Bᵩ = μ .* ψa .+ (1 .- μ) .* ψb .- ψϕ

    dμ = μ .* (1 .- μ)
    dϕ = ϕ

    ∇lₘ = ϕ .* (E1 - E2) .- ∇Bₘ
    ∇lᵩ = μ .* E1 .+ (1 .- μ) .* E2 .- ∇Bᵩ

    ∇²lₘ = -(ϕ.^2 .* (ψ1a .+ ψ1b))

    ∇²lᵩ = -(μ.^2 .* ψ1a .+
             (1 .- μ).^2 .* ψ1b .-
             ψ1ϕ)

    ∇²Bₘᵩ =
        ϕ .* μ .* ψ1a .+
        ψa .-
        ϕ .* (1 .- μ) .* ψ1b .-
        ψb

    ∇²lₘᵩ = E1 .- E2 .- ∇²Bₘᵩ

    Hmm = zeros(pm, pm)
    Hpp = zeros(pp, pp)
    Hmp = zeros(pm, pp)

    @inbounds for i in 1:T

        xm = @view Xm[i,:]
        xp = @view Xp[i,:]

        dμi  = dμ[i]
        dϕi  = dϕ[i]

        d2μi = dμi * (1 - 2μ[i])

        # βmβm block
        Hmm .+= (
            ∇²lₘ[i] * dμi^2 +
            ∇lₘ[i] * d2μi
        ) .* (xm * xm')

        # βpβp block
        Hpp .+= (
            ∇²lᵩ[i] * dϕi^2 +
            ∇lᵩ[i] * dϕi
        ) .* (xp * xp')

        # cross block
        Hmp .+= (
            ∇²lₘᵩ[i] * dμi * dϕi
        ) .* (xm * xp')
    end

    FInfo = -[Hmm Hmp;
              Hmp' Hpp] / T

    S = inv(sqrt(FInfo))

    return S
end


### Quick check
Xm = vcat(Xmean)
Xp = vcat(Xprec)
βm = [1,2,3]
βp = [1.0]
Info = 0
for i in 1:T
    Info = fisher_beta_blocks(Xmean, Xprec, β[i+1,:] , γ[i+1,:])
    println(inv(Info))
end

 