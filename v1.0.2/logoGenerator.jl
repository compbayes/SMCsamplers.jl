using Plots, Distributions, LaTeXStrings, Random, Measures

default()
gr(legend = nothing, grid = false, color = :blue, lw = 4, legendfontsize=14,
    xtickfontsize=14, ytickfontsize=14, xguidefontsize=20, yguidefontsize=20)

# official julia colors for blue, red, green, purple 
logo_colors = ["#4063D8", "#CB3C33", "#389826", "#9558B2"]   

Random.seed!(1234)
nTime = 4
nBalls = 4
nLines = 1
balls = zeros(nTime, nBalls)
μ = [1.0, 2.0, 3.0, 2.0]
for t in 1:nTime
    balls[t,:] = rand(Normal(μ[t]), nBalls)
end
ylims = [minimum(balls), maximum(balls)]
ylims[1] = ylims[1] - 0.1*(ylims[2]-ylims[1])
ylims[2] = ylims[2] + 0.1*(ylims[2]-ylims[1])

plt = plot(yaxis = false, xaxis = false, xlim = (0.5,nTime+1/2), ylim = ylims,
    background_color=:transparent, foreground_color=:black,
    size = (400, 400))
for colNumber in 1:nLines   
    plot!(plt, 1:nTime, balls[:,colNumber], lw = 2,
        color = logo_colors[colNumber])
end
for colNumber in 1:nBalls
    for t in 1:nTime
        if t == colNumber
            scatter!(plt, [t], [balls[t,colNumber]], 
                color = logo_colors[colNumber], 
                markersize = 16, markerstrokecolor = :auto, label = "") 
        else
            scatter!(plt, [t], [balls[t,colNumber]], 
                color = logo_colors[colNumber], 
                markersize = 6, markerstrokecolor = :auto, label = "") 
        end
    end
end
plt
savefig(dirname(@__FILE__)*"/assets/logo.svg")