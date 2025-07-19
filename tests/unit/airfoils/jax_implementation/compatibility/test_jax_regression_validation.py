"""
Regression testing for complex JAX airfoil operations.

This module provides regression testing to ensure that complex airfoil operations
maintain consistent behavior across different versions and configurations.

Requirements covered: 2.1, 8.2
"""

import json
from pathlib import Path
from typing import Any
from typing import Dict

import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.batch_processing import BatchAirfoilOps
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class RegressionTestData:
    """Utility class for managing regression test data."""

    def __init__(self, test_data_dir: str = "tests/TestData/regression"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

    def save_reference_data(self, test_name: str, data: Dict[str, Any]):
        """Save reference data for regression testing."""
        file_path = self.test_data_dir / f"{test_name}_reference.json"

        # Convert JAX arrays to lists for JSON serialization
        serializable_data = self._make_serializable(data)

        with open(file_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

    def load_reference_data(self, test_name: str) -> Dict[str, Any]:
        """Load reference data for regression testing."""
        file_path = self.test_data_dir / f"{test_name}_reference.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        # Convert lists back to JAX arrays
        return self._make_jax_arrays(data)

    def _make_serializable(self, obj):
        """Convert JAX arrays to serializable format."""
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def _make_jax_arrays(self, obj):
        """Convert serializable format back to JAX arrays."""
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float)):
            return jnp.array(obj)
        elif isinstance(obj, dict):
            return {k: self._make_jax_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_jax_arrays(item) for item in obj]
        else:
            return obj


class TestNACAGenerationRegression:
    """Test regression for NACA airfoil generation."""

    def setup_method(self):
        """Set up test data manager."""
        self.test_data = RegressionTestData()

    def test_naca4_generation_consistency(self):
        """Test that NACA 4-digit generation is consistent."""
        test_cases = [
            {"naca": "0012", "n_points": 100},
            {"naca": "2412", "n_points": 150},
            {"naca": "4415", "n_points": 200},
            {"naca": "6409", "n_points": 100},
        ]

        for case in test_cases:
            test_name = f"naca4_{case['naca']}_{case['n_points']}"

            # Generate airfoil
            airfoil = JaxAirfoil.naca4(case["naca"], n_points=case["n_points"])

            # Extract key properties
            current_data = {
                "coordinates": airfoil.get_coordinates(),
                "max_thickness": float(airfoil.max_thickness),
                "max_thickness_location": float(airfoil.max_thickness_location),
                "max_camber": float(airfoil.max_camber),
                "max_camber_location": float(airfoil.max_camber_location),
                "chord_length": float(airfoil.chord_length),
                "n_points": airfoil.n_points,
            }

            # Load reference data
            reference_data = self.test_data.load_reference_data(test_name)

            if reference_data is None:
                # First run - save reference data
                self.test_data.save_reference_data(test_name, current_data)
                pytest.skip(f"Saved reference data for {test_name}")
            else:
                # Compare with reference
                self._compare_airfoil_data(current_data, reference_data, test_name)

    def test_naca5_generation_consistency(self):
        """Test that NACA 5-digit generation is consistent."""
        test_cases = [
            {"naca": "23012", "n_points": 100},
            {"naca": "23015", "n_points": 150},
            {"naca": "44012", "n_points": 200},
        ]

        for case in test_cases:
            test_name = f"naca5_{case['naca']}_{case['n_points']}"

            # Generate airfoil
            airfoil = JaxAirfoil.naca5(case["naca"], n_points=case["n_points"])

            # Extract key properties
            current_data = {
                "coordinates": airfoil.get_coordinates(),
                "max_thickness": float(airfoil.max_thickness),
                "max_thickness_location": float(airfoil.max_thickness_location),
                "max_camber": float(airfoil.max_camber),
                "max_camber_location": float(airfoil.max_camber_location),
                "chord_length": float(airfoil.chord_length),
                "n_points": airfoil.n_points,
            }

            # Load reference data
            reference_data = self.test_data.load_reference_data(test_name)

            if reference_data is None:
                # First run - save reference data
                self.test_data.save_reference_data(test_name, current_data)
                pytest.skip(f"Saved reference data for {test_name}")
            else:
                # Compare with reference
                self._compare_airfoil_data(current_data, reference_data, test_name)

    def _compare_airfoil_data(self, current: Dict, reference: Dict, test_name: str):
        """Compare current airfoil data with reference."""
        # Compare scalar properties
        scalar_props = [
            "max_thickness",
            "max_thickness_location",
            "max_camber",
            "max_camber_location",
            "chord_length",
            "n_points",
        ]

        for prop in scalar_props:
            if prop == "n_points":
                assert current[prop] == reference[prop], f"{test_name}: {prop} mismatch"
            else:
                rel_error = abs(current[prop] - reference[prop]) / (
                    abs(reference[prop]) + 1e-10
                )
                assert (
                    rel_error < 1e-6
                ), f"{test_name}: {prop} regression, rel_error={rel_error:.2e}"

        # Compare coordinates
        current_coords = jnp.array(current["coordinates"])
        reference_coords = jnp.array(reference["coordinates"])

        coord_error = jnp.max(jnp.abs(current_coords - reference_coords))
        assert (
            coord_error < 1e-10
        ), f"{test_name}: coordinate regression, max_error={coord_error:.2e}"


class TestMorphingOperationRegression:
    """Test regression for airfoil morphing operations."""

    def setup_method(self):
        """Set up test data manager."""
        self.test_data = RegressionTestData()

    def test_morphing_consistency(self):
        """Test that morphing operations are consistent."""
        # Create base airfoils
        airfoil1 = JaxAirfoil.naca4("0012", n_points=100)
        airfoil2 = JaxAirfoil.naca4("4415", n_points=100)

        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for eta in eta_values:
            test_name = f"morphing_0012_4415_eta_{eta:.2f}"

            # Perform morphing
            morphed = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta)

            # Extract properties
            current_data = {
                "eta": eta,
                "coordinates": morphed.get_coordinates(),
                "max_thickness": float(morphed.max_thickness),
                "max_camber": float(morphed.max_camber),
                "chord_length": float(morphed.chord_length),
            }

            # Query surface at specific points
            query_x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            current_data.update(
                {
                    "thickness_at_query": morphed.thickness(query_x),
                    "camber_at_query": morphed.camber_line(query_x),
                    "upper_at_query": morphed.y_upper(query_x),
                    "lower_at_query": morphed.y_lower(query_x),
                },
            )

            # Load reference data
            reference_data = self.test_data.load_reference_data(test_name)

            if reference_data is None:
                # First run - save reference data
                self.test_data.save_reference_data(test_name, current_data)
                pytest.skip(f"Saved reference data for {test_name}")
            else:
                # Compare with reference
                self._compare_morphing_data(current_data, reference_data, test_name)

    def test_morphing_boundary_conditions(self):
        """Test morphing boundary conditions (eta=0 and eta=1)."""
        airfoil1 = JaxAirfoil.naca4("0008", n_points=150)
        airfoil2 = JaxAirfoil.naca4("2412", n_points=150)

        # Test eta=0 (should match airfoil1)
        morphed_0 = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 0.0)

        coords1 = jnp.array(airfoil1.get_coordinates())
        coords_morphed_0 = jnp.array(morphed_0.get_coordinates())

        coord_error_0 = jnp.max(jnp.abs(coords1 - coords_morphed_0))
        assert coord_error_0 < 1e-10, f"Morphing eta=0 error: {coord_error_0:.2e}"

        # Test eta=1 (should match airfoil2)
        morphed_1 = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 1.0)

        coords2 = jnp.array(airfoil2.get_coordinates())
        coords_morphed_1 = jnp.array(morphed_1.get_coordinates())

        coord_error_1 = jnp.max(jnp.abs(coords2 - coords_morphed_1))
        assert coord_error_1 < 1e-10, f"Morphing eta=1 error: {coord_error_1:.2e}"

    def _compare_morphing_data(self, current: Dict, reference: Dict, test_name: str):
        """Compare current morphing data with reference."""
        # Compare scalar properties
        scalar_props = ["eta", "max_thickness", "max_camber", "chord_length"]

        for prop in scalar_props:
            if prop == "eta":
                assert abs(current[prop] - reference[prop]) < 1e-10
            else:
                rel_error = abs(current[prop] - reference[prop]) / (
                    abs(reference[prop]) + 1e-10
                )
                assert (
                    rel_error < 1e-6
                ), f"{test_name}: {prop} regression, rel_error={rel_error:.2e}"

        # Compare arrays
        array_props = [
            "coordinates",
            "thickness_at_query",
            "camber_at_query",
            "upper_at_query",
            "lower_at_query",
        ]

        for prop in array_props:
            current_array = jnp.array(current[prop])
            reference_array = jnp.array(reference[prop])

            array_error = jnp.max(jnp.abs(current_array - reference_array))
            assert (
                array_error < 1e-10
            ), f"{test_name}: {prop} regression, max_error={array_error:.2e}"


class TestFlapOperationRegression:
    """Test regression for flap operations."""

    def setup_method(self):
        """Set up test data manager."""
        self.test_data = RegressionTestData()

    def test_flap_operation_consistency(self):
        """Test that flap operations are consistent."""
        base_airfoil = JaxAirfoil.naca4("2412", n_points=150)

        flap_configs = [
            {"hinge": 0.7, "angle": 10.0, "thickness_pos": 0.5, "extension": 1.0},
            {"hinge": 0.75, "angle": 20.0, "thickness_pos": 0.3, "extension": 1.2},
            {"hinge": 0.8, "angle": -15.0, "thickness_pos": 0.7, "extension": 0.9},
        ]

        for i, config in enumerate(flap_configs):
            test_name = f"flap_config_{i}"

            # Apply flap
            flapped = base_airfoil.flap(
                flap_hinge_chord_percentage=config["hinge"],
                flap_angle=config["angle"],
                flap_hinge_thickness_percentage=config["thickness_pos"],
                chord_extension=config["extension"],
            )

            # Extract properties
            current_data = {
                "config": config,
                "coordinates": flapped.get_coordinates(),
                "max_thickness": float(flapped.max_thickness),
                "max_camber": float(flapped.max_camber),
                "chord_length": float(flapped.chord_length),
            }

            # Query surface at specific points
            query_x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            current_data.update(
                {
                    "thickness_at_query": flapped.thickness(query_x),
                    "upper_at_query": flapped.y_upper(query_x),
                    "lower_at_query": flapped.y_lower(query_x),
                },
            )

            # Load reference data
            reference_data = self.test_data.load_reference_data(test_name)

            if reference_data is None:
                # First run - save reference data
                self.test_data.save_reference_data(test_name, current_data)
                pytest.skip(f"Saved reference data for {test_name}")
            else:
                # Compare with reference
                self._compare_flap_data(current_data, reference_data, test_name)

    def test_flap_zero_angle_consistency(self):
        """Test that zero flap angle produces unchanged airfoil."""
        base_airfoil = JaxAirfoil.naca4("0012", n_points=100)

        # Apply zero flap
        flapped = base_airfoil.flap(flap_hinge_chord_percentage=0.75, flap_angle=0.0)

        # Should be identical to original
        base_coords = jnp.array(base_airfoil.get_coordinates())
        flapped_coords = jnp.array(flapped.get_coordinates())

        coord_error = jnp.max(jnp.abs(base_coords - flapped_coords))
        assert coord_error < 1e-10, f"Zero flap angle error: {coord_error:.2e}"

    def _compare_flap_data(self, current: Dict, reference: Dict, test_name: str):
        """Compare current flap data with reference."""
        # Compare configuration
        assert current["config"] == reference["config"], f"{test_name}: config mismatch"

        # Compare scalar properties
        scalar_props = ["max_thickness", "max_camber", "chord_length"]

        for prop in scalar_props:
            rel_error = abs(current[prop] - reference[prop]) / (
                abs(reference[prop]) + 1e-10
            )
            assert (
                rel_error < 1e-6
            ), f"{test_name}: {prop} regression, rel_error={rel_error:.2e}"

        # Compare arrays
        array_props = [
            "coordinates",
            "thickness_at_query",
            "upper_at_query",
            "lower_at_query",
        ]

        for prop in array_props:
            current_array = jnp.array(current[prop])
            reference_array = jnp.array(reference[prop])

            array_error = jnp.max(jnp.abs(current_array - reference_array))
            assert (
                array_error < 1e-10
            ), f"{test_name}: {prop} regression, max_error={array_error:.2e}"


class TestRepanelingRegression:
    """Test regression for repaneling operations."""

    def setup_method(self):
        """Set up test data manager."""
        self.test_data = RegressionTestData()

    def test_repaneling_consistency(self):
        """Test that repaneling operations are consistent."""
        base_airfoil = JaxAirfoil.naca4("4415", n_points=100)

        target_points = [50, 150, 200]

        for n_points in target_points:
            test_name = f"repanel_4415_to_{n_points}"

            # Repanel airfoil
            repaneled = base_airfoil.repanel(n_points)

            # Extract properties
            current_data = {
                "target_points": n_points,
                "actual_points": repaneled.n_points,
                "coordinates": repaneled.get_coordinates(),
                "max_thickness": float(repaneled.max_thickness),
                "max_camber": float(repaneled.max_camber),
                "chord_length": float(repaneled.chord_length),
            }

            # Query surface at specific points to test interpolation quality
            query_x = jnp.linspace(0.05, 0.95, 20)
            current_data.update(
                {
                    "thickness_at_query": repaneled.thickness(query_x),
                    "camber_at_query": repaneled.camber_line(query_x),
                    "upper_at_query": repaneled.y_upper(query_x),
                    "lower_at_query": repaneled.y_lower(query_x),
                },
            )

            # Load reference data
            reference_data = self.test_data.load_reference_data(test_name)

            if reference_data is None:
                # First run - save reference data
                self.test_data.save_reference_data(test_name, current_data)
                pytest.skip(f"Saved reference data for {test_name}")
            else:
                # Compare with reference
                self._compare_repanel_data(current_data, reference_data, test_name)

    def test_repaneling_shape_preservation(self):
        """Test that repaneling preserves airfoil shape characteristics."""
        original_airfoil = JaxAirfoil.naca4("2412", n_points=200)

        # Repanel to different resolutions
        resolutions = [50, 100, 300, 500]

        for n_points in resolutions:
            repaneled = original_airfoil.repanel(n_points)

            # Key properties should be preserved within tolerance
            thickness_error = abs(
                repaneled.max_thickness - original_airfoil.max_thickness,
            )
            camber_error = abs(repaneled.max_camber - original_airfoil.max_camber)
            chord_error = abs(repaneled.chord_length - original_airfoil.chord_length)

            assert (
                thickness_error < 0.01
            ), f"Thickness preservation error at {n_points} points: {thickness_error}"
            assert (
                camber_error < 0.01
            ), f"Camber preservation error at {n_points} points: {camber_error}"
            assert (
                chord_error < 0.01
            ), f"Chord preservation error at {n_points} points: {chord_error}"

            # Surface queries should be similar
            query_x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
            orig_thickness = original_airfoil.thickness(query_x)
            repanel_thickness = repaneled.thickness(query_x)

            thickness_query_error = jnp.max(jnp.abs(orig_thickness - repanel_thickness))
            assert (
                thickness_query_error < 0.02
            ), f"Surface query error at {n_points} points: {thickness_query_error}"

    def _compare_repanel_data(self, current: Dict, reference: Dict, test_name: str):
        """Compare current repanel data with reference."""
        # Compare integer properties
        assert current["target_points"] == reference["target_points"]
        assert current["actual_points"] == reference["actual_points"]

        # Compare scalar properties
        scalar_props = ["max_thickness", "max_camber", "chord_length"]

        for prop in scalar_props:
            rel_error = abs(current[prop] - reference[prop]) / (
                abs(reference[prop]) + 1e-10
            )
            assert (
                rel_error < 1e-6
            ), f"{test_name}: {prop} regression, rel_error={rel_error:.2e}"

        # Compare arrays
        array_props = [
            "coordinates",
            "thickness_at_query",
            "camber_at_query",
            "upper_at_query",
            "lower_at_query",
        ]

        for prop in array_props:
            current_array = jnp.array(current[prop])
            reference_array = jnp.array(reference[prop])

            array_error = jnp.max(jnp.abs(current_array - reference_array))
            assert (
                array_error < 1e-10
            ), f"{test_name}: {prop} regression, max_error={array_error:.2e}"


class TestBatchOperationRegression:
    """Test regression for batch operations."""

    def setup_method(self):
        """Set up test data manager."""
        self.test_data = RegressionTestData()

    def test_batch_operation_consistency(self):
        """Test that batch operations are consistent."""
        # Create batch of different airfoils
        airfoils = [
            JaxAirfoil.naca4("0012", n_points=100),
            JaxAirfoil.naca4("2412", n_points=100),
            JaxAirfoil.naca4("4415", n_points=100),
            JaxAirfoil.naca5("23012", n_points=100),
        ]

        query_x = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])

        test_name = "batch_operations_mixed"

        # Perform batch operations
        batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)
        batch_camber = BatchAirfoilOps.batch_camber_line(airfoils, query_x)
        batch_upper = BatchAirfoilOps.batch_y_upper(airfoils, query_x)
        batch_lower = BatchAirfoilOps.batch_y_lower(airfoils, query_x)

        current_data = {
            "n_airfoils": len(airfoils),
            "query_points": len(query_x),
            "batch_thickness": batch_thickness,
            "batch_camber": batch_camber,
            "batch_upper": batch_upper,
            "batch_lower": batch_lower,
        }

        # Individual operations for comparison
        individual_thickness = jnp.array(
            [airfoil.thickness(query_x) for airfoil in airfoils],
        )
        individual_camber = jnp.array(
            [airfoil.camber_line(query_x) for airfoil in airfoils],
        )
        individual_upper = jnp.array([airfoil.y_upper(query_x) for airfoil in airfoils])
        individual_lower = jnp.array([airfoil.y_lower(query_x) for airfoil in airfoils])

        current_data.update(
            {
                "individual_thickness": individual_thickness,
                "individual_camber": individual_camber,
                "individual_upper": individual_upper,
                "individual_lower": individual_lower,
            },
        )

        # Load reference data
        reference_data = self.test_data.load_reference_data(test_name)

        if reference_data is None:
            # First run - save reference data
            self.test_data.save_reference_data(test_name, current_data)
            pytest.skip(f"Saved reference data for {test_name}")
        else:
            # Compare with reference
            self._compare_batch_data(current_data, reference_data, test_name)

        # Also verify batch vs individual consistency
        self._verify_batch_individual_consistency(current_data)

    def _compare_batch_data(self, current: Dict, reference: Dict, test_name: str):
        """Compare current batch data with reference."""
        # Compare metadata
        assert current["n_airfoils"] == reference["n_airfoils"]
        assert current["query_points"] == reference["query_points"]

        # Compare arrays
        array_props = [
            "batch_thickness",
            "batch_camber",
            "batch_upper",
            "batch_lower",
            "individual_thickness",
            "individual_camber",
            "individual_upper",
            "individual_lower",
        ]

        for prop in array_props:
            current_array = jnp.array(current[prop])
            reference_array = jnp.array(reference[prop])

            array_error = jnp.max(jnp.abs(current_array - reference_array))
            assert (
                array_error < 1e-10
            ), f"{test_name}: {prop} regression, max_error={array_error:.2e}"

    def _verify_batch_individual_consistency(self, data: Dict):
        """Verify that batch operations match individual operations."""
        batch_thickness = jnp.array(data["batch_thickness"])
        individual_thickness = jnp.array(data["individual_thickness"])

        thickness_error = jnp.max(jnp.abs(batch_thickness - individual_thickness))
        assert (
            thickness_error < 1e-12
        ), f"Batch-individual thickness mismatch: {thickness_error:.2e}"

        batch_camber = jnp.array(data["batch_camber"])
        individual_camber = jnp.array(data["individual_camber"])

        camber_error = jnp.max(jnp.abs(batch_camber - individual_camber))
        assert (
            camber_error < 1e-12
        ), f"Batch-individual camber mismatch: {camber_error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
