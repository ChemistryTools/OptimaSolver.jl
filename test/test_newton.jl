@testset "Newton step" begin
    # Simple 2×4 system: same water/acid conservation as canonicalizer test
    A = Float64[
        2  1  1  2
        1  0  1  0
    ]
    m, ns = size(A)
    can = Canonicalizer(A)

    h = [2.0, 3.0, 1.5, 4.0]     # positive Hessian diagonal
    ex = [0.1, -0.2, 0.05, -0.1]  # optimality residual
    ew = [0.01, -0.005]            # feasibility residual

    ws = NewtonStep(ns, m)
    dn, dy = compute_step!(ws, can, h, ex, ew)

    # Verify KKT: [ H Aᵀ; A 0 ] [dn; dy] = [-ex; -ew]
    lhs_primal = h .* dn .+ A' * dy
    lhs_dual = A * dn
    @test lhs_primal ≈ -ex atol = 1.0e-10
    @test lhs_dual ≈ -ew atol = 1.0e-10

    # clamp_step: no step crosses lower bound
    n = [1.0, 0.5, 2.0, 0.1]
    lb = zeros(4)
    dn_test = [-0.9, 0.1, -0.5, -0.05]
    α = clamp_step(n, lb, dn_test)
    @test α > 0
    @test all(n .+ α .* dn_test .> lb)

    # Step with τ=0.995: should stay strictly inside
    α2 = clamp_step(n, lb, dn_test; τ = 0.995)
    @test all(n .+ α2 .* dn_test .> lb .+ 1.0e-15)
end
