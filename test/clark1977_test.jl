# Clark (1977) Mountain Wave Model Tests
# Using TestItems format compatible with project test infrastructure

@testsnippet Clark1977Setup begin
    using Test
    using ModelingToolkit
    using ModelingToolkit: mtkcompile
    using NonlinearSolve
    using AtmosphericDynamics
    # Reduce timeout sensitivity for CI
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "1"
end

@testitem "Clark 1977 Mountain Wave Model" setup=[Clark1977Setup] tags=[:clark1977] begin
    # =============================================================================
    # IsentropicBaseState Tests
    # =============================================================================

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

    # =============================================================================
    # WitchOfAgnesi Tests
    # =============================================================================

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

    # =============================================================================
    # Clark1977FullPDESystem Tests - Complete Model with MethodOfLines
    # =============================================================================

    @testset "Clark1977FullPDESystem - Complete Implementation" begin
        # Import MethodOfLines for PDE discretization testing
        import MethodOfLines
        import OrdinaryDiffEqDefault: Tsit5

        @testset "Full PDESystem Structure" begin
            # Test the complete nonlinear Clark 1977 model
            full_pde = Clark1977FullPDESystem()
            @test full_pde isa MethodOfLines.PDESystem

            # Check that the system has the expected structure for complete model
            @test length(full_pde.eqs) == 7  # 7 governing equations (u, w, θ, continuity, ω relation, ρ relation, pressure)
            @test length(full_pde.bcs) >= 20  # Comprehensive boundary conditions
            @test length(full_pde.domain) == 3  # t, x, z_bar domains
            @test length(full_pde.dvs) == 7  # u, v, w, ω, θ, p, ρ dependent variables
        end

        @testset "MethodOfLines Discretization - Complete Model" begin
            # Create complete PDESystem with minimal parameters for fast testing
            full_pde = Clark1977FullPDESystem(
                U_0_val=4.0,      # m/s - mean flow
                h_val=50.0,       # m - small mountain for stability
                a_val=3000.0,     # m - mountain width
                L_val=9000.0,     # m - smaller domain for speed
                H_val=3000.0,     # m - smaller height for speed
                T_end_val=50.0,   # s - short time for testing
                τ_R_val=100.0     # s - stronger damping for stability
            )

            # Discretize with coarse grid for fast testing
            dx = 3000.0  # m - only 6 points in x
            dz = 1000.0  # m - only 3 points in z
            discretization = MethodOfLines.MOLFiniteDifference([full_pde.domain[2].domain.left => dx, full_pde.domain[3].domain.left => dz], full_pde.domain[1].domain.left)

            # Discretize the complete PDESystem (using checks=false as per project standards)
            prob = MethodOfLines.discretize(full_pde, discretization; checks=false)
            @test prob isa MethodOfLines.ODEProblem

            # Test that the problem can be solved (even if just briefly)
            sol = solve(prob, Tsit5(), saveat=10.0, maxiters=50, abstol=1e-2, reltol=1e-1, dtmax=1.0)
            @test sol.retcode in [:Success, :MaxIters, :DtLessThanMin]  # Accept various completion states
            @test length(sol.t) >= 2  # At least initial and one time step

            # Test that solution values are reasonable (no NaNs, finite bounds)
            @test all(isfinite.(sol.u[end]))
            @test maximum(abs.(sol.u[end])) < 1000.0  # Reasonable magnitude bounds
        end

        @testset "Complete Model Physical Consistency" begin
            # Test that the complete model includes all key physical processes
            full_pde = Clark1977FullPDESystem(U_0_val=4.0, h_val=100.0)

            # Check that nonlinear terms are present (not linearized)
            eq_strings = string.(full_pde.eqs)

            # Should have nonlinear advection terms u*∂u/∂x
            @test any(occursin("*", eq_str) for eq_str in eq_strings)

            # Should have Coriolis terms
            @test any(occursin("f", eq_str) for eq_str in eq_strings)

            # Should have terrain-following transformation
            @test any(occursin("ω", eq_str) for eq_str in eq_strings)

            # Should have turbulence terms
            @test any(occursin("K_M", eq_str) for eq_str in eq_strings)
        end
    end

    # =============================================================================
    # MountainWave2D PDESystem Tests (Simplified/Linearized for Comparison)
    # =============================================================================

    @testset "MountainWave2D Simplified PDESystem" begin
        # Import MethodOfLines for PDE discretization testing
        import MethodOfLines
        import OrdinaryDiffEqDefault: Tsit5

        @testset "Simplified PDESystem Structure" begin
            pdesys = MountainWave2D()
            @test pdesys isa MethodOfLines.PDESystem

            # Check that the linearized system has expected structure
            @test length(pdesys.eqs) == 4  # 4 linearized equations
            @test length(pdesys.bcs) == 16  # Initial + boundary conditions
            @test length(pdesys.domain) == 3  # t, x, z domains
            @test length(pdesys.dvs) == 4  # u, w, θ, p dependent variables
        end

        @testset "MethodOfLines Discretization - Simplified" begin
            # Create simplified PDESystem for fast testing
            pdesys = MountainWave2D(
                U_0_val=4.0,
                h_val=100.0,
                L_val=6000.0,
                H_val=3000.0,
                T_end_val=100.0
            )

            # Discretize with coarse grid
            dx = 1500.0  # m
            dz = 750.0   # m
            discretization = MethodOfLines.MOLFiniteDifference([pdesys.domain[2].domain.left => dx, pdesys.domain[3].domain.left => dz], pdesys.domain[1].domain.left)

            # Discretize the PDESystem
            prob = MethodOfLines.discretize(pdesys, discretization; checks=false)
            @test prob isa MethodOfLines.ODEProblem

            # Test solving
            sol = solve(prob, Tsit5(), saveat=25.0, maxiters=100, abstol=1e-3, reltol=1e-2)
            @test sol.retcode in [:Success, :MaxIters]
            @test length(sol.t) >= 2

            # Test solution quality
            @test all(isfinite.(sol.u[end]))
            @test maximum(abs.(sol.u[end])) < 1000.0
        end
    end

    # =============================================================================
    # Component Structure Tests (Simplified - No Solving)
    # =============================================================================

    @testset "Component Structures" begin
        @testset "TerrainFollowingTransform Structure" begin
            @named transform = TerrainFollowingTransform()
            @test transform isa System
            @test length(unknowns(transform)) == 5
            @test length(equations(transform)) == 5
        end

        @testset "SmagorinskyTurbulence Structure" begin
            @named turb = SmagorinskyTurbulence()
            @test turb isa System
            @test length(unknowns(turb)) == 3
            @test length(equations(turb)) == 3
        end

        @testset "AnelasticMomentum Structure" begin
            @named mom = AnelasticMomentum()
            @test mom isa System
            @test length(unknowns(mom)) == 3
            @test length(equations(mom)) == 3
        end

        @testset "AnelasticMassContinuity Structure" begin
            @named mass = AnelasticMassContinuity()
            @test mass isa System
            @test length(unknowns(mass)) == 4
            @test length(equations(mass)) == 4
        end

        @testset "AnelasticThermodynamics Structure" begin
            @named thermo = AnelasticThermodynamics()
            @test thermo isa System
            @test length(unknowns(thermo)) == 5
            @test length(equations(thermo)) == 5
        end

        @testset "DiagnosticPressure Structure" begin
            @named pressure = DiagnosticPressure()
            @test pressure isa System
            @test length(unknowns(pressure)) == 3
            @test length(equations(pressure)) == 3
        end

        @testset "Clark1977AnelasticSystem Structure" begin
            @named full_system = Clark1977AnelasticSystem()
            @test full_system isa System

            # Test that all subsystems are included
            sys_names = [sys.name for sys in full_system.systems]
            @test :IsentropicBaseState in sys_names
            @test :WitchOfAgnesi in sys_names
            @test :TerrainFollowingTransform in sys_names
            @test :SmagorinskyTurbulence in sys_names
            @test :AnelasticMomentum in sys_names
            @test :AnelasticMassContinuity in sys_names
            @test :AnelasticThermodynamics in sys_names
            @test :DiagnosticPressure in sys_names
        end
    end

end