@testset "SMCutils.jl" begin

    # Testing the splitEqualGroups function for different group sizes and data lengths
    T = 10
    nPerGroup = 3
    Y, Z, groupSizes = splitEqualGroups(1:T, [rand(T, 2)], nPerGroup)

    @test groupSizes[end] == rem(T, nPerGroup)
    @test length(Y[1]) == size(Z[1], 1) == nPerGroup
    @test length(Y[end]) == size(Z[end], 1) == rem(T, nPerGroup)
    @test length(Y) == ceil(Int, T/nPerGroup)
    @test length(Y[end]) == rem(T, nPerGroup)

    # Testing when there is a single covariate entered as a matrix
    T = 10
    nPerGroup = 4
    Y, Z, groupSizes = splitEqualGroups(1:T, rand(T, 1), nPerGroup)
    @test length(Y[1]) == size(Z[1], 1) == nPerGroup

    # Testing when there is a single covariate entered as a vector
    Y, Z, groupSizes = splitEqualGroups(1:T, rand(T), nPerGroup)
    @test length(Y[1]) == size(Z[1], 1) == nPerGroup

    # Testing when there is a single covariate entered as a vector
    Y, Z, groupSizes = splitEqualGroups(1:T, [], nPerGroup)
    @test isempty(Z)

end