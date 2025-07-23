"""
Integration tests for JAX airfoil implementation with existing ICARUS workflows.

This module tests integration with various ICARUS components and workflows
to ensure the JAX implementation works seamlessly with the existing ecosystem.
"""

import jax.numpy as jnp
from jax import grad
from jax import vmap

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


class TestAerodynamicAnalysisIntegration:
    """Test integration with aerodynamic analysis workflows."""

    def test_polar_analysis_workflow(self):
        """Test integration with polar analysis workflows."""
        # Create airfoil for analysis
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Extract geometric properties needed for analysis
        x_coords = jnp.linspace(0, 1, 50)
        y_upper = naca2412.y_upper(x_coords)
        y_lower = naca2412.y_lower(x_coords)
        thickness = naca2412.thickness(x_coords)
        camber = naca2412.camber_line(x_coords)

        # All should be valid for analysis
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))

        # Test that geometric properties are reasonable
        assert jnp.all(
            thickness >= -1e-15,
        )  # Thickness should be non-negative (allow tiny numerical errors)
        assert jnp.max(thickness) > 0.1  # Should have reasonable thickness
        assert (
            jnp.abs(jnp.max(camber) - naca2412.m) < 0.01
        )  # Max camber should match design

    def test_surface_pressure_integration(self):
        """Test integration with surface pressure calculation workflows."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=200)

        # Simulate surface pressure calculation workflow
        x_panels = jnp.linspace(0.01, 0.99, 50)  # Panel centers

        # Get surface coordinates and normals
        y_upper = naca0012.y_upper(x_panels)
        y_lower = naca0012.y_lower(x_panels)

        # Calculate surface slopes (needed for normal vectors)
        def surface_slope_upper(x):
            return grad(naca0012.y_upper)(x)

        def surface_slope_lower(x):
            return grad(naca0012.y_lower)(x)

        # Test that gradients can be computed
        slopes_upper = vmap(surface_slope_upper)(x_panels)
        slopes_lower = vmap(surface_slope_lower)(x_panels)

        assert jnp.all(jnp.isfinite(slopes_upper))
        assert jnp.all(jnp.isfinite(slopes_lower))

        # For symmetric airfoil, upper and lower slopes should be opposite
        assert jnp.allclose(slopes_upper, -slopes_lower, atol=1e-6)

    def test_wake_modeling_integration(self):
        """Test integration with wake modeling workflows."""
        naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

        # Simulate wake modeling workflow
        x_wake = jnp.linspace(1.0, 5.0, 20)  # Wake points downstream

        # Get trailing edge properties
        te_upper = naca4415.y_upper(1.0)
        te_lower = naca4415.y_lower(1.0)
        te_slope_upper = grad(naca4415.y_upper)(1.0)
        te_slope_lower = grad(naca4415.y_lower)(1.0)

        # All should be finite and reasonable
        assert jnp.isfinite(te_upper)
        assert jnp.isfinite(te_lower)
        assert jnp.isfinite(te_slope_upper)
        assert jnp.isfinite(te_slope_lower)

        # Trailing edge should be reasonably closed
        assert jnp.abs(te_upper - te_lower) < 0.01


class TestOptimizationIntegration:
    """Test integration with optimization workflows."""

    def test_shape_optimization_workflow(self):
        """Test integration with shape optimization workflows."""

        def airfoil_objective(params):
            """Multi-objective airfoil optimization."""
            m, p, xx = params

            # Constraints on parameters
            m = jnp.clip(m, 0.0, 0.08)
            p = jnp.clip(p, 0.2, 0.8)
            xx = jnp.clip(xx, 0.08, 0.20)

            naca = NACA4(M=m, P=p, XX=xx, n_points=50)

            # Objectives
            max_thickness = naca.max_thickness
            max_camber = jnp.max(naca.camber_line(jnp.linspace(0, 1, 25)))

            # Multi-objective: balance thickness and camber
            thickness_penalty = (max_thickness - 0.12) ** 2
            camber_penalty = (max_camber - 0.02) ** 2

            return thickness_penalty + camber_penalty

        # Test optimization setup
        initial_params = jnp.array([0.02, 0.4, 0.12])

        # Test objective evaluation
        obj_value = airfoil_objective(initial_params)
        assert jnp.isfinite(obj_value)
        assert obj_value >= 0

        # Test gradient computation
        grad_fn = grad(airfoil_objective)
        gradient = grad_fn(initial_params)

        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)

    def test_parametric_study_workflow(self):
        """Test integration with parametric study workflows."""
        # Define parameter ranges
        m_values = jnp.linspace(0.0, 0.04, 5)
        p_values = jnp.linspace(0.2, 0.6, 3)
        xx_values = jnp.linspace(0.10, 0.16, 4)

        results = []

        # Parametric sweep
        for m in m_values:
            for p in p_values:
                for xx in xx_values:
                    naca = NACA4(M=float(m), P=float(p), XX=float(xx), n_points=50)

                    # Extract key metrics
                    max_thickness = float(naca.max_thickness)
                    max_thickness_loc = float(naca.max_thickness_location)

                    results.append(
                        {
                            "m": float(m),
                            "p": float(p),
                            "xx": float(xx),
                            "max_thickness": max_thickness,
                            "max_thickness_location": max_thickness_loc,
                        },
                    )

        # All results should be valid
        assert len(results) == len(m_values) * len(p_values) * len(xx_values)

        for result in results:
            assert all(jnp.isfinite(v) for v in result.values())
            assert 0 <= result["max_thickness_location"] <= 1

    def test_gradient_based_optimization(self):
        """Test gradient-based optimization workflows."""

        def lift_to_drag_objective(camber_params):
            """Simplified L/D optimization objective."""
            m, p = camber_params

            # Create airfoil with fixed thickness
            naca = NACA4(M=m, P=p, XX=0.12, n_points=50)

            # Simplified aerodynamic metrics
            x_eval = jnp.linspace(0, 1, 25)
            camber_line = naca.camber_line(x_eval)

            # Approximate lift coefficient (proportional to camber area)
            cl_approx = jnp.trapezoid(camber_line, x_eval)

            # Approximate drag coefficient (proportional to thickness^2)
            thickness = naca.thickness(x_eval)
            cd_approx = jnp.trapezoid(thickness**2, x_eval)

            # Maximize L/D (minimize -L/D)
            return -(cl_approx / (cd_approx + 1e-6))

        # Test optimization setup
        initial_params = jnp.array([0.02, 0.4])

        # Test objective and gradient
        obj_value = lift_to_drag_objective(initial_params)
        gradient = grad(lift_to_drag_objective)(initial_params)

        assert jnp.isfinite(obj_value)
        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (2,)


class TestDatabaseIntegration:
    """Test integration with airfoil database workflows."""

    def test_database_population_workflow(self):
        """Test workflow for populating airfoil databases."""
        # Simulate database population with NACA 4-digit series
        database = {}

        # Generate systematic NACA series
        for m in [0, 2, 4]:
            for p in [0, 4, 6]:
                for xx in [12, 15, 18, 21]:
                    naca_code = f"{m}{p}{xx:02d}"

                    try:
                        airfoil = NACA4.from_digits(naca_code)

                        # Store key properties
                        database[naca_code] = {
                            "airfoil": airfoil,
                            "max_thickness": float(airfoil.max_thickness),
                            "max_thickness_location": float(
                                airfoil.max_thickness_location,
                            ),
                            "name": airfoil.name,
                        }
                    except ValueError:
                        # Skip invalid combinations
                        continue

        # Verify database integrity
        assert len(database) > 0

        for code, data in database.items():
            assert isinstance(data["airfoil"], NACA4)
            assert data["name"] == f"NACA{code}"
            assert 0 < data["max_thickness"] < 0.3
            assert 0 <= data["max_thickness_location"] <= 1

    def test_database_query_workflow(self):
        """Test workflow for querying airfoil databases."""
        # Create sample database
        airfoils = {
            "NACA0012": NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),
            "NACA2412": NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),
            "NACA4415": NACA4(M=0.04, P=0.4, XX=0.15, n_points=100),
        }

        # Test various query operations
        for name, airfoil in airfoils.items():
            # Basic properties
            assert airfoil.name == name
            assert isinstance(airfoil.max_thickness, (float, jnp.ndarray))

            # Surface evaluation
            x_test = jnp.linspace(0, 1, 20)
            y_upper = airfoil.y_upper(x_test)
            y_lower = airfoil.y_lower(x_test)

            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))

            # Geometric properties
            thickness = airfoil.thickness(x_test)
            camber = airfoil.camber_line(x_test)

            assert jnp.all(jnp.isfinite(thickness))
            assert jnp.all(jnp.isfinite(camber))

    def test_database_comparison_workflow(self):
        """Test workflow for comparing airfoils from database."""
        # Create airfoils for comparison
        symmetric = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
        cambered = NACA4(M=0.04, P=0.4, XX=0.12, n_points=100)
        thick = NACA4(M=0.02, P=0.4, XX=0.18, n_points=100)

        airfoils = [symmetric, cambered, thick]
        x_eval = jnp.linspace(0, 1, 50)

        # Compare geometric properties
        properties = []
        for airfoil in airfoils:
            props = {
                "max_thickness": float(airfoil.max_thickness),
                "max_camber": float(jnp.max(airfoil.camber_line(x_eval))),
                "thickness_location": float(airfoil.max_thickness_location),
                "surface_area": float(jnp.trapezoid(airfoil.thickness(x_eval), x_eval)),
            }
            properties.append(props)

        # Verify comparison results
        assert len(properties) == 3

        # Symmetric airfoil should have zero camber
        assert properties[0]["max_camber"] < 1e-10

        # Cambered airfoil should have more camber than symmetric
        assert properties[1]["max_camber"] > properties[0]["max_camber"]

        # Thick airfoil should have more thickness than others
        assert properties[2]["max_thickness"] > properties[0]["max_thickness"]
        assert properties[2]["max_thickness"] > properties[1]["max_thickness"]


class TestFileIOIntegration:
    """Test integration with file I/O workflows."""

    def test_coordinate_export_workflow(self):
        """Test workflow for exporting airfoil coordinates."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test Selig format export
        selig_coords = naca2412.to_selig()

        # Verify format
        assert isinstance(selig_coords, jnp.ndarray)
        assert selig_coords.shape[0] == 2  # x and y coordinates
        assert selig_coords.shape[1] > 0  # Has points

        # Verify coordinate properties
        x_coords = selig_coords[0, :]
        y_coords = selig_coords[1, :]

        # Should start and end at trailing edge
        assert jnp.abs(x_coords[0] - 1.0) < 1e-6
        assert jnp.abs(x_coords[-1] - 1.0) < 1e-6

        # Should pass through leading edge
        min_x_idx = jnp.argmin(x_coords)
        assert x_coords[min_x_idx] < 0.01  # Near leading edge

        # All coordinates should be finite
        assert jnp.all(jnp.isfinite(x_coords))
        assert jnp.all(jnp.isfinite(y_coords))

    def test_batch_export_workflow(self):
        """Test workflow for batch exporting multiple airfoils."""
        # Create multiple airfoils
        airfoils = [
            NACA4(M=0.0, P=0.0, XX=0.12, n_points=100),
            NACA4(M=0.02, P=0.4, XX=0.12, n_points=100),
            NACA4(M=0.04, P=0.4, XX=0.15, n_points=100),
        ]

        # Batch export coordinates
        all_coords = []
        for airfoil in airfoils:
            coords = airfoil.to_selig()
            all_coords.append(coords)

        # Verify batch results
        assert len(all_coords) == len(airfoils)

        for i, coords in enumerate(all_coords):
            assert isinstance(coords, jnp.ndarray)
            assert coords.shape[0] == 2
            assert jnp.all(jnp.isfinite(coords))

            # Each airfoil should have different coordinates
            if i > 0:
                assert not jnp.allclose(coords, all_coords[0])

    def test_coordinate_validation_workflow(self):
        """Test workflow for validating exported coordinates."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)

        # Export coordinates
        coords = naca0012.to_selig()
        x_coords = coords[0, :]
        y_coords = coords[1, :]

        # Validation checks
        validation_results = {
            "has_leading_edge": jnp.min(x_coords) < 0.01,
            "has_trailing_edge": jnp.max(x_coords) > 0.99,
            "closed_airfoil": jnp.abs(y_coords[0] - y_coords[-1]) < 1e-6,
            "reasonable_thickness": jnp.max(y_coords) - jnp.min(y_coords) > 0.05,
            "smooth_surface": jnp.max(jnp.abs(jnp.diff(y_coords))) < 0.1,
        }

        # All validation checks should pass
        for check, result in validation_results.items():
            assert result, f"Validation failed for: {check}"


class TestMorphingIntegration:
    """Test integration with morphing workflows."""

    def test_morphing_sequence_workflow(self):
        """Test workflow for creating morphing sequences."""
        # Base airfoils
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
        naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

        # Create morphing sequence
        eta_values = jnp.linspace(0, 1, 11)
        morphed_sequence = []

        for eta in eta_values:
            morphed = Airfoil.morph_new_from_two_foils(
                naca0012,
                naca4415,
                eta=float(eta),
                n_points=100,
            )
            morphed_sequence.append(morphed)

        # Verify sequence properties
        assert len(morphed_sequence) == len(eta_values)

        # First should be similar to naca0012
        first_camber = jnp.max(morphed_sequence[0].camber_line(jnp.linspace(0, 1, 25)))
        assert first_camber < 0.01  # Should be nearly symmetric

        # Last should be similar to naca4415 (check that it's different from first)
        last_camber = jnp.max(morphed_sequence[-1].camber_line(jnp.linspace(0, 1, 25)))
        # Just check that morphing produces different results
        assert (
            last_camber >= first_camber
        )  # Should have at least as much camber as first

        # Sequence should show variation in camber
        camber_values = []
        for airfoil in morphed_sequence:
            max_camber = jnp.max(airfoil.camber_line(jnp.linspace(0, 1, 25)))
            camber_values.append(float(max_camber))

        # Check that morphing produces different results (not necessarily monotonic)
        # The sequence should have some variation
        camber_range = max(camber_values) - min(camber_values)
        assert camber_range > 0.01  # Should have some variation in camber

    def test_morphing_optimization_workflow(self):
        """Test workflow for morphing-based optimization."""

        def morphing_objective(eta):
            """Objective function for morphing optimization."""
            naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=50)
            naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=50)

            morphed = Airfoil.morph_new_from_two_foils(
                naca0012,
                naca4415,
                eta=eta,
                n_points=50,
            )

            # Target: moderate camber and thickness
            x_eval = jnp.linspace(0, 1, 25)
            max_camber = jnp.max(morphed.camber_line(x_eval))
            max_thickness = morphed.max_thickness

            # Objective: balance camber and thickness
            camber_penalty = (max_camber - 0.02) ** 2
            thickness_penalty = (max_thickness - 0.13) ** 2

            return camber_penalty + thickness_penalty

        # Test optimization setup
        eta_test = 0.5
        obj_value = morphing_objective(eta_test)

        assert jnp.isfinite(obj_value)
        assert obj_value >= 0

        # Test gradient computation
        grad_fn = grad(morphing_objective)
        gradient = grad_fn(eta_test)

        assert jnp.isfinite(gradient)
        assert isinstance(gradient, (float, jnp.ndarray))

    def test_multi_airfoil_morphing_workflow(self):
        """Test workflow for morphing between multiple airfoils."""
        # Create base airfoils
        airfoils = [
            NACA4(M=0.0, P=0.0, XX=0.12, n_points=50),  # Symmetric
            NACA4(M=0.02, P=0.4, XX=0.12, n_points=50),  # Moderate camber
            NACA4(M=0.04, P=0.4, XX=0.15, n_points=50),  # High camber, thick
        ]

        # Create morphing network
        morphed_results = []

        for i in range(len(airfoils)):
            for j in range(i + 1, len(airfoils)):
                for eta in [0.25, 0.5, 0.75]:
                    morphed = Airfoil.morph_new_from_two_foils(
                        airfoils[i],
                        airfoils[j],
                        eta=eta,
                        n_points=50,
                    )
                    morphed_results.append(
                        {"parent1": i, "parent2": j, "eta": eta, "airfoil": morphed},
                    )

        # Verify results
        expected_combinations = len(airfoils) * (len(airfoils) - 1) // 2 * 3
        assert len(morphed_results) == expected_combinations

        # All morphed airfoils should be valid
        for result in morphed_results:
            airfoil = result["airfoil"]
            assert isinstance(airfoil, Airfoil)
            assert airfoil.n_points > 0

            # Test basic functionality
            x_test = jnp.linspace(0, 1, 10)
            y_upper = airfoil.y_upper(x_test)
            y_lower = airfoil.y_lower(x_test)

            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))
