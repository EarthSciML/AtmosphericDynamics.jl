using Test
using ModelingToolkit
using ModelingToolkit: mtkcompile
using NonlinearSolve
using AtmosphericDynamics

#=============================================================================
# Clark (1977) Mountain Wave Model Tests
=============================================================================#

@testset "Clark 1977 Mountain Wave Model" begin

    #=============================================================================
    # IsentropicBaseState Tests
    =============================================================================#

    @testset "IsentropicBaseState" begin

        @testset "Structure" begin
            @named bs = IsentropicBaseState()
            @test bs isa System
            @test length(unknowns(bs)) == 4
            @test length(equations(bs)) == 4
        end

        @testset "Surface Values" begin
            @named bs = IsentropicBaseState()
            sys = mtkcompile(bs)

            prob = NonlinearProblem(sys, Dict())
            sol = solve(prob, NewtonRaphson())

            @test sol[sys.T_bar] ≈ 300.0 rtol = 1.0e-10
            @test sol[sys.p_bar] ≈ 1.0e5 rtol = 1.0e-10
            @test sol[sys.ρ_bar] ≈ 1.0e5 / (287.0 * 300.0) rtol = 1.0e-6
            @test sol[sys.H_s] ≈ 1004.0 * 300.0 / 9.81 rtol = 1.0e-6
        end

        @testset "Profile" begin
            @named bs = IsentropicBaseState()
            sys = mtkcompile(bs)

            prob = NonlinearProblem(sys, Dict(sys.z => 5000.0))
            sol = solve(prob, NewtonRaphson())

            H_s = 1004.0 * 300.0 / 9.81
            κ_val = 287.0 / 1004.0

            @test sol[sys.T_bar] ≈ 300.0 * (1 - 5000.0 / H_s) rtol = 1.0e-6
            @test sol[sys.p_bar] ≈ 1.0e5 * (1 - 5000.0 / H_s)^(1 / κ_val) rtol = 1.0e-4
            @test sol[sys.ρ_bar] ≈ 1.0e5 / (287.0 * 300.0) * (1 - 5000.0 / H_s)^(1 / κ_val - 1) rtol = 1.0e-4
            @test sol[sys.T_bar] < 300.0
            @test sol[sys.p_bar] < 1.0e5
        end

        @testset "Hydrostatic Balance" begin
            @named bs = IsentropicBaseState()
            sys = mtkcompile(bs)

            sol1 = solve(NonlinearProblem(sys, Dict(sys.z => 0.0)), NewtonRaphson())
            sol2 = solve(NonlinearProblem(sys, Dict(sys.z => 100.0)), NewtonRaphson())

            dp_dz = (sol2[sys.p_bar] - sol1[sys.p_bar]) / 100.0
            ρ_avg = (sol1[sys.ρ_bar] + sol2[sys.ρ_bar]) / 2
            @test dp_dz ≈ -ρ_avg * 9.81 rtol = 1.0e-3
        end

    end

    #=============================================================================
    # WitchOfAgnesi Tests
    =============================================================================#

    @testset "WitchOfAgnesi" begin

        @testset "Structure" begin
            @named topo = WitchOfAgnesi()
            @test topo isa System
            @test length(unknowns(topo)) == 2
            @test length(equations(topo)) == 2
        end

        @testset "Peak and Symmetry" begin
            @named topo = WitchOfAgnesi()
            sys = mtkcompile(topo)

            sol_peak = solve(NonlinearProblem(sys, Dict(sys.x => 0.0)), NewtonRaphson())
            @test sol_peak[sys.z_s] ≈ 100.0 rtol = 1.0e-10
            @test sol_peak[sys.dz_s_dx] ≈ 0.0 atol = 1.0e-10

            sol_half = solve(NonlinearProblem(sys, Dict(sys.x => 3000.0)), NewtonRaphson())
            @test sol_half[sys.z_s] ≈ 50.0 rtol = 1.0e-10

            sol_neg = solve(NonlinearProblem(sys, Dict(sys.x => -3000.0)), NewtonRaphson())
            @test sol_neg[sys.z_s] ≈ sol_half[sys.z_s] rtol = 1.0e-10
            @test sol_neg[sys.dz_s_dx] ≈ -sol_half[sys.dz_s_dx] rtol = 1.0e-10
        end

        @testset "Limiting Behavior" begin
            @named topo = WitchOfAgnesi()
            sys = mtkcompile(topo)

            sol_far = solve(NonlinearProblem(sys, Dict(sys.x => 100000.0)), NewtonRaphson())
            @test sol_far[sys.z_s] < 1.0
            @test abs(sol_far[sys.dz_s_dx]) < 1.0e-4
        end

        @testset "1km Mountain" begin
            @named topo = WitchOfAgnesi()
            sys = mtkcompile(topo)

            sol = solve(NonlinearProblem(sys, Dict(sys.h_mtn => 1000.0, sys.x => 0.0)), NewtonRaphson())
            @test sol[sys.z_s] ≈ 1000.0 rtol = 1.0e-10
        end

    end

    #=============================================================================
    # TerrainFollowingTransform Tests
    =============================================================================#

    @testset "TerrainFollowingTransform" begin

        @testset "Structure" begin
            @named tft = TerrainFollowingTransform()
            @test tft isa System
            @test length(unknowns(tft)) == 5
            @test length(equations(tft)) == 5
        end

        @testset "Flat Terrain" begin
            @named tft = TerrainFollowingTransform()
            sys = mtkcompile(tft)

            sol = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => 0.0, sys.z_bar => 4000.0, sys.H => 8000.0
                    )
                ), NewtonRaphson()
            )

            @test sol[sys.G_sqrt] ≈ 1.0 rtol = 1.0e-10
            @test sol[sys.G_sqrt_G13] ≈ 0.0 atol = 1.0e-10
            @test sol[sys.G_sqrt_G23] ≈ 0.0 atol = 1.0e-10
            @test sol[sys.z_phys] ≈ 4000.0 rtol = 1.0e-10
        end

        @testset "Boundary Mapping" begin
            @named tft = TerrainFollowingTransform()
            sys = mtkcompile(tft)

            z_s_val = 100.0
            H_val = 8000.0

            sol_bot = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => z_s_val, sys.z_bar => 0.0, sys.H => H_val
                    )
                ), NewtonRaphson()
            )
            @test sol_bot[sys.z_phys] ≈ z_s_val rtol = 1.0e-10

            sol_top = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => z_s_val, sys.z_bar => H_val, sys.H => H_val
                    )
                ), NewtonRaphson()
            )
            @test sol_top[sys.z_phys] ≈ H_val rtol = 1.0e-10
        end

        @testset "Jacobian" begin
            @named tft = TerrainFollowingTransform()
            sys = mtkcompile(tft)

            sol = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => 100.0, sys.H => 8000.0
                    )
                ), NewtonRaphson()
            )
            @test sol[sys.G_sqrt] ≈ 1.0 - 100.0 / 8000.0 rtol = 1.0e-10
        end

        @testset "Metric Coefficients" begin
            @named tft = TerrainFollowingTransform()
            sys = mtkcompile(tft)

            H_val = 8000.0
            dz_s_dx_val = -0.01

            sol_surf = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => 100.0, sys.z_bar => 0.0, sys.H => H_val,
                        sys.dz_s_dx => dz_s_dx_val
                    )
                ), NewtonRaphson()
            )
            @test sol_surf[sys.G_sqrt_G13] ≈ (0.0 / H_val - 1) * dz_s_dx_val rtol = 1.0e-10

            sol_top = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.z_s => 100.0, sys.z_bar => H_val, sys.H => H_val,
                        sys.dz_s_dx => dz_s_dx_val
                    )
                ), NewtonRaphson()
            )
            @test sol_top[sys.G_sqrt_G13] ≈ 0.0 atol = 1.0e-10
        end

    end

    #=============================================================================
    # SmagorinskyTurbulence Tests
    =============================================================================#

    @testset "SmagorinskyTurbulence" begin

        @testset "Structure" begin
            @named smag = SmagorinskyTurbulence()
            @test smag isa System
            @test length(unknowns(smag)) == 3
            @test length(equations(smag)) == 3
        end

        @testset "Zero Deformation" begin
            @named smag = SmagorinskyTurbulence()
            sys = mtkcompile(smag)

            sol = solve(NonlinearProblem(sys, Dict()), NewtonRaphson())
            @test sol[sys.Def] ≈ 0.0 atol = 1.0e-10
            @test sol[sys.K_M] ≈ 0.0 atol = 1.0e-10
            @test sol[sys.K_H] ≈ 0.0 atol = 1.0e-10
        end

        @testset "Known Values" begin
            @named smag = SmagorinskyTurbulence()
            sys = mtkcompile(smag)

            sol = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.D13 => 0.01, sys.Δ_grid => 600.0
                    )
                ), NewtonRaphson()
            )
            @test sol[sys.Def] ≈ 0.01 rtol = 1.0e-6
            @test sol[sys.K_M] ≈ (0.25 * 600.0)^2 * 0.01 rtol = 1.0e-6
            @test sol[sys.K_H] ≈ sol[sys.K_M] rtol = 1.0e-10
        end

        @testset "Full Deformation" begin
            @named smag = SmagorinskyTurbulence()
            sys = mtkcompile(smag)

            D11_val, D22_val, D33_val = 0.001, -0.001, 0.0
            D12_val, D13_val, D23_val = 0.005, 0.003, 0.002

            sol = solve(
                NonlinearProblem(
                    sys, Dict(
                        sys.D11 => D11_val, sys.D22 => D22_val, sys.D33 => D33_val,
                        sys.D12 => D12_val, sys.D13 => D13_val, sys.D23 => D23_val,
                        sys.Δ_grid => 600.0
                    )
                ), NewtonRaphson()
            )

            Def_expected = sqrt(
                0.5 * (D11_val^2 + D22_val^2 + D33_val^2) +
                    D12_val^2 + D13_val^2 + D23_val^2
            )
            @test sol[sys.Def] ≈ Def_expected rtol = 1.0e-6
            @test sol[sys.K_M] > 0.0
        end

    end

    #=============================================================================
    # MountainWave2D Tests
    =============================================================================#

    @testset "MountainWave2D" begin

        @testset "Construction" begin
            sys = MountainWave2D()
            @test sys isa ModelingToolkit.PDESystem
            @test length(sys.eqs) == 4
            @test length(sys.bcs) == 20
        end

        @testset "Froude Number" begin
            # Inverse Froude number F = a*N/(2π*U) (Eq. 7.5)
            F = 3000.0 * 0.01 / (2π * 4.0)
            @test F ≈ 1.18 rtol = 0.02
        end

        @testset "Solve" begin
            using MethodOfLines
            using OrdinaryDiffEqDefault
            using SciMLBase

            # Very coarse grid and short simulation — MOL discretization is O(N^4)
            sys = MountainWave2D(
                L_val = 6000.0, H_val = 4000.0, T_end_val = 50.0,
                h_val = 100.0, a_val = 3000.0
            )

            dx = 2000.0  # 6 x-points
            dz = 1000.0  # 4 z-points

            x_var = sys.ivs[2]
            z_var = sys.ivs[3]

            discretization = MOLFiniteDifference(
                [x_var => dx, z_var => dz], sys.ivs[1]
            )
            prob = MethodOfLines.discretize(sys, discretization; checks = false)
            sol = solve(prob; saveat = 50.0)

            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

    end

end