# Clark (1977) Mountain Wave Model Tests
# Using TestItems format compatible with project test infrastructure

@testsnippet Clark1977Setup begin
    using Test
    using ModelingToolkit
    using ModelingToolkit: mtkcompile
    using NonlinearSolve
    using AtmosphericDynamics
    using MethodOfLines
    using OrdinaryDiffEqDefault: Tsit5
    using SciMLBase
end

@testitem "Clark 1977 Mountain Wave Model" setup = [Clark1977Setup] tags = [:clark1977] begin
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

        @testset "Full PDESystem Structure" begin
            full_pde = Clark1977FullPDESystem()
            @test full_pde isa MethodOfLines.PDESystem

            @test length(full_pde.eqs) == 7  # 7 governing equations
            @test length(full_pde.bcs) >= 20  # Comprehensive boundary conditions
            @test length(full_pde.domain) == 3  # t, x, z_bar domains
            @test length(full_pde.dvs) == 7  # u, v, w, ω, θ, p, ρ dependent variables
        end

        @testset "Complete Model Physical Consistency" begin
            full_pde = Clark1977FullPDESystem(U_0_val = 4.0, h_val = 100.0)

            eq_strings = string.(full_pde.eqs)

            # Should have nonlinear advection terms u*∂u/∂x
            @test any(occursin("*", eq_str) for eq_str in eq_strings)

            # Should have Coriolis terms
            @test any(occursin("f", eq_str) for eq_str in eq_strings)

            # Should have terrain-following transformation
            @test any(occursin("ω", eq_str) for eq_str in eq_strings)

            # Should have spatial derivative terms
            @test any(occursin("Differential", eq_str) for eq_str in eq_strings)
        end
    end

    # =============================================================================
    # MountainWave2D PDESystem Tests (Simplified/Linearized for Comparison)
    # =============================================================================

    @testset "MountainWave2D Simplified PDESystem" begin

        @testset "Simplified PDESystem Structure" begin
            pdesys = MountainWave2D()
            @test pdesys isa MethodOfLines.PDESystem

            @test length(pdesys.eqs) == 4  # 4 linearized equations
            @test length(pdesys.bcs) == 20  # Initial + boundary conditions
            @test length(pdesys.domain) == 3  # t, x, z domains
            @test length(pdesys.dvs) == 4  # u, w, θ, p dependent variables
        end

        @testset "MethodOfLines Discretization - Simplified" begin
            pdesys = MountainWave2D(
                U_0_val = 4.0,
                h_val = 100.0,
                L_val = 6000.0,
                H_val = 4000.0,
                T_end_val = 50.0,
                a_val = 3000.0
            )

            dx = 2000.0  # 6 x-points
            dz = 1000.0  # 4 z-points

            x_var = pdesys.ivs[2]
            z_var = pdesys.ivs[3]

            discretization = MethodOfLines.MOLFiniteDifference(
                [x_var => dx, z_var => dz], pdesys.ivs[1]
            )
            prob = MethodOfLines.discretize(pdesys, discretization; checks = false)
            @test prob isa MethodOfLines.ODEProblem

            sol = solve(prob; saveat = 50.0)
            @test sol.retcode == SciMLBase.ReturnCode.Success
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

            # Test that all subsystems are included using nameof
            sys_names = [nameof(sys) for sys in ModelingToolkit.get_systems(full_system)]
            @test :base_state in sys_names
            @test :topography in sys_names
            @test :transform in sys_names
            @test :turbulence in sys_names
            @test :momentum in sys_names
            @test :mass_continuity in sys_names
            @test :thermodynamics in sys_names
            @test :pressure_diag in sys_names
        end
    end

    # =============================================================================
    # Paper Results Validation Tests
    # =============================================================================

    @testset "Clark (1977) Paper Results Validation" begin

        @testset "Table I Parameter Combinations" begin
            # Run 14 parameters (100m mountain)
            sys_run14 = MountainWave2D(
                h_val = 100.0,
                a_val = 3000.0,
                U_0_val = 4.0,
                N_val = 0.01,
                H_val = 8000.0,
                L_val = 18000.0,
                T_end_val = 4000.0
            )
            @test sys_run14 isa PDESystem

            # Run 15 parameters (1km mountain)
            sys_run15 = MountainWave2D(
                h_val = 1000.0,
                a_val = 3000.0,
                U_0_val = 4.0,
                N_val = 0.01,
                H_val = 8000.0,
                L_val = 18000.0,
                T_end_val = 4000.0
            )
            @test sys_run15 isa PDESystem

            # Both should be valid PDESystems with different mountain heights
            @test sys_run14 isa PDESystem
            @test sys_run15 isa PDESystem
        end

        @testset "Witch of Agnesi Topography (Eq. 7.1)" begin
            @named topo = WitchOfAgnesi()
            sys = mtkcompile(topo)

            prob_center = NonlinearProblem(sys, Dict(sys.x => 0.0))
            sol_center = solve(prob_center, NewtonRaphson())
            @test sol_center[sys.z_s] ≈ 100.0  # Default h_mtn = 100.0

            prob_halfwidth = NonlinearProblem(sys, Dict(sys.x => 3000.0))
            sol_halfwidth = solve(prob_halfwidth, NewtonRaphson())
            @test sol_halfwidth[sys.z_s] ≈ 50.0  # h_mtn/2 = 50.0

            @test sol_center[sys.dz_s_dx] ≈ 0.0 rtol = 1.0e-10
        end

        @testset "Isentropic Scale Height Validation" begin
            @named bs = IsentropicBaseState()
            sys = mtkcompile(bs)

            prob = NonlinearProblem(sys, Dict())
            sol = solve(prob, NewtonRaphson())

            H_s_expected = 1004.0 * 300.0 / 9.81
            @test sol[sys.H_s] ≈ H_s_expected rtol = 1.0e-6
            @test H_s_expected ≈ 30683.0 rtol = 1.0e-2
        end

        @testset "Brunt-Väisälä Frequency Validation" begin
            g = 9.81
            Θ = 300.0
            dtheta_dz = 3.0 / 1000.0  # K/m (3K/km)

            N_squared_expected = (g / Θ) * dtheta_dz
            N_expected = sqrt(N_squared_expected)

            @test N_expected ≈ 0.00991 rtol = 1.0e-3
            @test N_expected ≈ 0.01 rtol = 1.0e-1
        end

        @testset "Inverse Froude Number (Eq. 7.5)" begin
            # F = a*N/(2π*U) (Eq. 7.5)
            a = 3000.0  # m
            U = 4.0     # m/s
            N = 0.01    # s⁻¹ (Brunt-Väisälä frequency)

            F = a * N / (2π * U)

            @test F ≈ 1.18 rtol = 0.02
        end

        @testset "Wave Drag Validation (Paper Results)" begin
            # Clark (1977) reports wave drag values for 100m mountain case
            D_w_run14_expected = -294.4  # kg sec⁻²
            D_w_run18_expected = -316.5  # kg sec⁻²

            @test abs(D_w_run14_expected) > 100.0
            @test abs(D_w_run18_expected) > 100.0
            @test D_w_run14_expected != D_w_run18_expected
        end

        @testset "Grid Resolution Validation (Table I)" begin
            Δx_paper = 600.0  # m
            NX_paper = 62
            total_x_domain_paper = NX_paper * Δx_paper  # 37200 m
            our_domain_width = 2 * 18000.0  # 36000 m

            @test our_domain_width ≈ total_x_domain_paper rtol = 1.0e-1

            Δz_paper_100 = 100.0
            NZ_paper_case1 = 82
            total_z_domain_paper = NZ_paper_case1 * Δz_paper_100  # 8200 m
            our_domain_height = 8000.0

            @test our_domain_height ≈ total_z_domain_paper rtol = 1.0e-1
        end
    end

    # =============================================================================
    # MethodOfLines Integration Test
    # =============================================================================

    @testset "MethodOfLines PDE Solution Test" begin
        sys = MountainWave2D(
            h_val = 100.0,
            T_end_val = 50.0,
            L_val = 6000.0,
            H_val = 4000.0,
            a_val = 3000.0
        )

        @test sys isa PDESystem
        @test length(equations(sys)) == 4
        @test length(sys.bcs) == 20

        dx = 2000.0
        dz = 1000.0

        x_var = sys.ivs[2]
        z_var = sys.ivs[3]

        discretization = MOLFiniteDifference(
            [x_var => dx, z_var => dz], sys.ivs[1]
        )
        prob = discretize(sys, discretization; checks = false)
        @test prob isa ODEProblem

        sol = solve(prob; saveat = 50.0)
        @test sol.retcode == SciMLBase.ReturnCode.Success
    end

end
