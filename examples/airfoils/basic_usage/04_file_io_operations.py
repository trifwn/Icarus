#!/usr/bin/env python3
"""
File I/O Operations Examples

This example demonstrates file input/output operations with JAX airfoils:
1. Loading airfoils from coordinate files
2. Saving airfoils in different formats
3. Working with standard airfoil file formats
4. Batch file operations
5. Data validation and error handling
6. Integration with airfoil databases

The JAX implementation maintains compatibility with standard airfoil file formats
while providing enhanced computational capabilities.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


def create_sample_airfoil_files():
    """Create sample airfoil files for demonstration purposes."""
    print("=== Creating Sample Airfoil Files ===")

    # Create a temporary directory for our examples
    temp_dir = Path(tempfile.mkdtemp(prefix="airfoil_geometry_examples_"))
    print(f"Creating sample files in: {temp_dir}")

    # Sample 1: NACA 0012 in Selig format (TE to TE)
    naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
    selig_coords = naca0012.to_selig()

    selig_file = temp_dir / "naca0012_selig.dat"
    with open(selig_file, "w") as f:
        f.write("NACA 0012\\n")
        for i in range(selig_coords.shape[1]):
            f.write(f"{selig_coords[0, i]:10.6f} {selig_coords[1, i]:10.6f}\\n")

    print(f"Created Selig format file: {selig_file.name}")

    # Sample 2: NACA 2412 in Lednicer format (upper then lower)
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)
    upper = naca2412.upper_surface
    lower = naca2412.lower_surface

    lednicer_file = temp_dir / "naca2412_lednicer.dat"
    with open(lednicer_file, "w") as f:
        f.write("NACA 2412\\n")
        f.write(f"{upper.shape[1]} {lower.shape[1]}\\n")

        # Upper surface (LE to TE)
        for i in range(upper.shape[1]):
            f.write(f"{upper[0, i]:10.6f} {upper[1, i]:10.6f}\\n")

        # Lower surface (LE to TE)
        for i in range(lower.shape[1]):
            f.write(f"{lower[0, i]:10.6f} {lower[1, i]:10.6f}\\n")

    print(f"Created Lednicer format file: {lednicer_file.name}")

    # Sample 3: Simple coordinate file
    simple_file = temp_dir / "simple_airfoil.dat"
    with open(simple_file, "w") as f:
        f.write("Simple Symmetric Airfoil\\n")
        # Create simple coordinates
        x = np.linspace(0, 1, 21)
        y_upper = 0.1 * np.sin(np.pi * x)
        y_lower = -0.1 * np.sin(np.pi * x)

        # Write in Selig format
        for i in range(len(x) - 1, -1, -1):  # Upper surface (TE to LE)
            f.write(f"{x[i]:8.5f} {y_upper[i]:8.5f}\\n")
        for i in range(1, len(x)):  # Lower surface (LE to TE)
            f.write(f"{x[i]:8.5f} {y_lower[i]:8.5f}\\n")

    print(f"Created simple format file: {simple_file.name}")

    # Sample 4: Noisy data file (for error handling demonstration)
    noisy_file = temp_dir / "noisy_airfoil.dat"
    with open(noisy_file, "w") as f:
        f.write("Noisy Airfoil Data\\n")
        f.write("# This is a comment line\\n")
        f.write("\\n")  # Empty line

        # Add some valid data with noise
        x = np.linspace(0, 1, 15)
        y = 0.08 * np.sin(np.pi * x) + 0.01 * np.random.randn(len(x))

        for i in range(len(x)):
            if i == 5:  # Add a bad line
                f.write("bad_data_line\\n")
            else:
                f.write(f"{x[i]:8.5f} {y[i]:8.5f}\\n")

    print(f"Created noisy data file: {noisy_file.name}")

    return temp_dir


def demonstrate_file_loading():
    """Demonstrate loading airfoils from various file formats."""
    print("\\n=== Loading Airfoils from Files ===")

    # Create sample files
    temp_dir = create_sample_airfoil_files()

    loaded_airfoils = []

    # Load Selig format file
    try:
        selig_file = temp_dir / "naca0012_selig.dat"
        airfoil_selig = Airfoil.from_file(str(selig_file))
        loaded_airfoils.append(airfoil_selig)
        print(f"Successfully loaded: {airfoil_selig.name}")
        print(f"  Points: {airfoil_selig.n_points}")
        print(f"  Max thickness: {airfoil_selig.max_thickness:.4f}")
    except Exception as e:
        print(f"Error loading Selig file: {e}")

    # Load simple format file
    try:
        simple_file = temp_dir / "simple_airfoil.dat"
        airfoil_simple = Airfoil.from_file(str(simple_file))
        loaded_airfoils.append(airfoil_simple)
        print(f"Successfully loaded: {airfoil_simple.name}")
        print(f"  Points: {airfoil_simple.n_points}")
        print(f"  Max thickness: {airfoil_simple.max_thickness:.4f}")
    except Exception as e:
        print(f"Error loading simple file: {e}")

    # Demonstrate error handling with noisy file
    try:
        noisy_file = temp_dir / "noisy_airfoil.dat"
        airfoil_noisy = Airfoil.from_file(str(noisy_file))
        loaded_airfoils.append(airfoil_noisy)
        print(f"Successfully loaded (with filtering): {airfoil_noisy.name}")
        print(f"  Points: {airfoil_noisy.n_points}")
    except Exception as e:
        print(f"Error loading noisy file: {e}")

    return loaded_airfoils, temp_dir


def demonstrate_file_saving():
    """Demonstrate saving airfoils in different formats."""
    print("\\n=== Saving Airfoils to Files ===")

    # Create airfoils to save
    naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

    # Create temporary directory for output
    temp_dir = Path(tempfile.mkdtemp(prefix="airfoil_geometry_output_"))
    print(f"Saving files to: {temp_dir}")

    # Method 1: Save in Selig format (TE to TE)
    selig_output = temp_dir / f"{naca4415.name}_selig.dat"
    save_airfoil_selig_format(naca4415, selig_output)
    print(f"Saved Selig format: {selig_output.name}")

    # Method 2: Save upper and lower surfaces separately
    upper_output = temp_dir / f"{naca4415.name}_upper.dat"
    lower_output = temp_dir / f"{naca4415.name}_lower.dat"
    save_airfoil_surfaces(naca4415, upper_output, lower_output)
    print(f"Saved surfaces: {upper_output.name}, {lower_output.name}")

    # Method 3: Save with metadata
    metadata_output = temp_dir / f"{naca4415.name}_with_metadata.dat"
    save_airfoil_with_metadata(naca4415, metadata_output)
    print(f"Saved with metadata: {metadata_output.name}")

    # Method 4: Save in CSV format for spreadsheet compatibility
    csv_output = temp_dir / f"{naca4415.name}.csv"
    save_airfoil_csv_format(naca4415, csv_output)
    print(f"Saved CSV format: {csv_output.name}")

    return temp_dir


def save_airfoil_selig_format(airfoil, filename):
    """Save airfoil in Selig format (TE to TE)."""
    selig_coords = airfoil.to_selig()

    with open(filename, "w") as f:
        f.write(f"{airfoil.name}\\n")
        for i in range(selig_coords.shape[1]):
            f.write(f"{selig_coords[0, i]:12.8f} {selig_coords[1, i]:12.8f}\\n")


def save_airfoil_surfaces(airfoil, upper_filename, lower_filename):
    """Save upper and lower surfaces separately."""
    upper = airfoil.upper_surface
    lower = airfoil.lower_surface

    # Save upper surface
    with open(upper_filename, "w") as f:
        f.write(f"{airfoil.name} - Upper Surface\\n")
        for i in range(upper.shape[1]):
            f.write(f"{upper[0, i]:12.8f} {upper[1, i]:12.8f}\\n")

    # Save lower surface
    with open(lower_filename, "w") as f:
        f.write(f"{airfoil.name} - Lower Surface\\n")
        for i in range(lower.shape[1]):
            f.write(f"{lower[0, i]:12.8f} {lower[1, i]:12.8f}\\n")


def save_airfoil_with_metadata(airfoil, filename):
    """Save airfoil with comprehensive metadata."""
    with open(filename, "w") as f:
        # Write header with metadata
        f.write(f"# Airfoil: {airfoil.name}\\n")
        f.write("# Generated using JAX Airfoil Implementation\\n")
        f.write(f"# Number of points: {airfoil.n_points}\\n")
        f.write(f"# Maximum thickness: {airfoil.max_thickness:.6f}\\n")
        f.write(f"# Max thickness location: {airfoil.max_thickness_location:.6f}\\n")

        # Add NACA-specific parameters if available
        if hasattr(airfoil, "m") and hasattr(airfoil, "p") and hasattr(airfoil, "xx"):
            f.write(
                f"# NACA Parameters: M={airfoil.m:.3f}, P={airfoil.p:.3f}, XX={airfoil.xx:.3f}\\n",
            )

        f.write("# Format: Selig (TE to TE)\\n")
        f.write("#\\n")
        f.write(f"{airfoil.name}\\n")

        # Write coordinates
        selig_coords = airfoil.to_selig()
        for i in range(selig_coords.shape[1]):
            f.write(f"{selig_coords[0, i]:12.8f} {selig_coords[1, i]:12.8f}\\n")


def save_airfoil_csv_format(airfoil, filename):
    """Save airfoil in CSV format for spreadsheet compatibility."""
    upper = airfoil.upper_surface
    lower = airfoil.lower_surface

    with open(filename, "w") as f:
        f.write(f"# {airfoil.name}\\n")
        f.write("x_upper,y_upper,x_lower,y_lower\\n")

        # Pad shorter surface with NaN if needed
        max_points = max(upper.shape[1], lower.shape[1])

        for i in range(max_points):
            if i < upper.shape[1]:
                x_u, y_u = upper[0, i], upper[1, i]
            else:
                x_u, y_u = float("nan"), float("nan")

            if i < lower.shape[1]:
                x_l, y_l = lower[0, i], lower[1, i]
            else:
                x_l, y_l = float("nan"), float("nan")

            f.write(f"{x_u:.8f},{y_u:.8f},{x_l:.8f},{y_l:.8f}\\n")


def batch_file_operations():
    """Demonstrate batch operations on multiple files."""
    print("\\n=== Batch File Operations ===")

    # Create multiple NACA airfoils
    naca_specs = [
        ("0009", 0.0, 0.0, 0.09),
        ("0012", 0.0, 0.0, 0.12),
        ("2412", 0.02, 0.4, 0.12),
        ("4415", 0.04, 0.4, 0.15),
        ("6409", 0.06, 0.4, 0.09),
    ]

    # Create temporary directory
    batch_dir = Path(tempfile.mkdtemp(prefix="airfoil_geometry_batch_"))
    print(f"Batch processing directory: {batch_dir}")

    airfoils = []

    # Generate and save multiple airfoils
    for name, m, p, xx in naca_specs:
        # Create airfoil
        airfoil = NACA4(M=m, P=p, XX=xx, n_points=100)
        airfoils.append(airfoil)

        # Save in multiple formats
        base_name = f"naca{name}"

        # Selig format
        selig_file = batch_dir / f"{base_name}_selig.dat"
        save_airfoil_selig_format(airfoil, selig_file)

        # CSV format
        csv_file = batch_dir / f"{base_name}.csv"
        save_airfoil_csv_format(airfoil, csv_file)

        print(
            f"Processed {airfoil.name}: saved as {base_name}_selig.dat and {base_name}.csv",
        )

    # Create summary file
    summary_file = batch_dir / "airfoil_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Airfoil Batch Processing Summary\\n")
        f.write("=" * 40 + "\\n")
        f.write(f"Generated: {len(airfoils)} airfoils\\n")
        f.write(f"Date: {np.datetime64('now')}\\n\\n")

        f.write("Airfoil Properties:\\n")
        f.write("Name      Max t/c   t/c loc   Camber   Camber loc\\n")
        f.write("-" * 50 + "\\n")

        for airfoil in airfoils:
            max_t = airfoil.max_thickness
            max_t_loc = airfoil.max_thickness_location
            camber = float(airfoil.m) if hasattr(airfoil, "m") else 0.0
            camber_loc = float(airfoil.p) if hasattr(airfoil, "p") else 0.0

            f.write(
                f"{airfoil.name:<8} {max_t:7.4f}   {max_t_loc:7.4f}   {camber:6.3f}    {camber_loc:6.3f}\\n",
            )

    print(f"Created summary file: {summary_file.name}")

    return airfoils, batch_dir


def data_validation_examples():
    """Demonstrate data validation and error handling."""
    print("\\n=== Data Validation and Error Handling ===")

    def validate_airfoil_data(filename):
        """Validate airfoil data from file."""
        print(f"Validating: {Path(filename).name}")

        try:
            # Attempt to load airfoil
            airfoil = Airfoil.from_file(filename)

            # Basic validation checks
            checks = {
                "loaded_successfully": True,
                "has_points": airfoil.n_points > 0,
                "reasonable_thickness": 0.01 < airfoil.max_thickness < 0.5,
                "closed_airfoil": True,  # Assume from_file handles this
                "monotonic_x": True,  # Check if x-coordinates are reasonable
            }

            # Check x-coordinate monotonicity for upper surface
            upper = airfoil.upper_surface
            x_upper = upper[0]
            checks["monotonic_x"] = bool(jnp.all(jnp.diff(x_upper) >= -1e-10))

            # Report validation results
            print("  Validation results:")
            for check, result in checks.items():
                status = "✓" if result else "✗"
                print(f"    {status} {check}: {result}")

            if all(checks.values()):
                print("  ✓ All validation checks passed")
                return airfoil, True
            else:
                print("  ⚠ Some validation checks failed")
                return airfoil, False

        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            return None, False

    # Create test files with various issues
    test_dir = Path(tempfile.mkdtemp(prefix="airfoil_geometry_validation_"))

    # Good file
    good_airfoil = NACA4(M=0.02, P=0.4, XX=0.12, n_points=50)
    good_file = test_dir / "good_airfoil.dat"
    save_airfoil_selig_format(good_airfoil, good_file)

    # File with too few points
    few_points_file = test_dir / "few_points.dat"
    with open(few_points_file, "w") as f:
        f.write("Few Points Airfoil\\n")
        f.write("1.0 0.0\\n")
        f.write("0.0 0.0\\n")
        f.write("1.0 0.0\\n")

    # File with unrealistic thickness
    thick_file = test_dir / "unrealistic_thick.dat"
    with open(thick_file, "w") as f:
        f.write("Unrealistically Thick Airfoil\\n")
        x = np.linspace(0, 1, 21)
        y_upper = 0.8 * np.sin(np.pi * x)  # 80% thickness!
        y_lower = -0.8 * np.sin(np.pi * x)

        # Selig format
        for i in range(len(x) - 1, -1, -1):
            f.write(f"{x[i]:8.5f} {y_upper[i]:8.5f}\\n")
        for i in range(1, len(x)):
            f.write(f"{x[i]:8.5f} {y_lower[i]:8.5f}\\n")

    # Validate all test files
    test_files = [good_file, few_points_file, thick_file]
    results = []

    for test_file in test_files:
        airfoil, is_valid = validate_airfoil_data(test_file)
        results.append((test_file.name, airfoil, is_valid))

    return results


def integration_with_databases():
    """Demonstrate integration with airfoil databases (conceptual)."""
    print("\\n=== Integration with Airfoil Databases (Conceptual) ===")

    print("Conceptual database integration examples:")
    print("")

    print("1. UIUC Airfoil Database Integration:")
    print("   # Download airfoil from UIUC database")
    print("   airfoil = Airfoil.from_uiuc_database('naca0012')")
    print("   airfoil.save_selig_format('naca0012_uiuc.dat')")
    print("")

    print("2. Custom Database Operations:")
    print("   # Save to custom database")
    print("   database = AirfoilDatabase('my_airfoils.db')")
    print("   database.add_airfoil(naca2412)")
    print("   database.add_metadata(naca2412, {'source': 'generated', 'date': '2024'})")
    print("")

    print("3. Batch Database Operations:")
    print("   # Load multiple airfoils from database")
    print("   airfoils = database.load_airfoils_by_criteria(")
    print("       thickness_range=(0.10, 0.15),")
    print("       camber_range=(0.0, 0.04)")
    print("   )")
    print("")

    print("4. Format Conversion:")
    print("   # Convert between different formats")
    print("   converter = AirfoilFormatConverter()")
    print("   converter.selig_to_lednicer('input.dat', 'output.dat')")
    print("   converter.dat_to_csv('airfoil.dat', 'airfoil.csv')")


def plot_file_operations_results():
    """Create visualizations of file operation results."""
    print("\\n=== Visualizing File Operations Results ===")

    # Load some airfoils for plotting
    loaded_airfoils, _ = demonstrate_file_loading()

    if loaded_airfoils:
        plt.figure(figsize=(15, 10))

        # Plot 1: Loaded airfoils comparison
        plt.subplot(2, 3, 1)
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, airfoil in enumerate(loaded_airfoils[:5]):  # Limit to 5 airfoils
            upper = airfoil.upper_surface
            lower = airfoil.lower_surface
            color = colors[i % len(colors)]

            plt.plot(
                upper[0],
                upper[1],
                color=color,
                linewidth=2,
                label=f"{airfoil.name} upper",
            )
            plt.plot(
                lower[0],
                lower[1],
                color=color,
                linewidth=2,
                linestyle="--",
                label=f"{airfoil.name} lower",
            )

        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.title("Loaded Airfoils")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot 2: File format comparison (conceptual)
        plt.subplot(2, 3, 2)
        formats = ["Selig", "Lednicer", "CSV", "Custom"]
        file_sizes = [1.2, 1.5, 2.1, 1.8]  # Conceptual file sizes in KB

        bars = plt.bar(formats, file_sizes, color=["blue", "green", "red", "orange"])
        plt.ylabel("File Size (KB)")
        plt.title("File Format Comparison")
        plt.xticks(rotation=45)

        for bar, size in zip(bars, file_sizes):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{size:.1f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Data validation results
        plt.subplot(2, 3, 3)
        validation_categories = ["Valid Files", "Invalid Files", "Corrupted Files"]
        validation_counts = [8, 2, 1]  # Example counts
        colors_val = ["green", "orange", "red"]

        plt.pie(
            validation_counts,
            labels=validation_categories,
            colors=colors_val,
            autopct="%1.1f%%",
        )
        plt.title("Data Validation Results")

        # Plot 4: Processing workflow
        plt.subplot(2, 3, 4)
        workflow_steps = ["Load", "Validate", "Process", "Save"]
        processing_times = [0.1, 0.05, 0.2, 0.08]  # Example times in seconds

        plt.plot(workflow_steps, processing_times, "o-", linewidth=2, markersize=8)
        plt.ylabel("Processing Time (s)")
        plt.title("File Processing Workflow")
        plt.grid(True, alpha=0.3)

        # Plot 5: Batch operation summary
        plt.subplot(2, 3, 5)
        batch_airfoils, _ = batch_file_operations()

        names = [airfoil.name for airfoil in batch_airfoils]
        thicknesses = [airfoil.max_thickness for airfoil in batch_airfoils]

        plt.bar(names, thicknesses, color="skyblue")
        plt.ylabel("Maximum Thickness")
        plt.title("Batch Processed Airfoils")
        plt.xticks(rotation=45)

        # Plot 6: File size vs complexity
        plt.subplot(2, 3, 6)
        n_points = [50, 100, 200, 500, 1000]
        file_sizes_est = [0.8, 1.5, 3.0, 7.5, 15.0]  # Estimated file sizes in KB

        plt.loglog(n_points, file_sizes_est, "o-", linewidth=2, markersize=8)
        plt.xlabel("Number of Points")
        plt.ylabel("File Size (KB)")
        plt.title("File Size vs Complexity")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating all file I/O operations."""
    print("JAX Airfoil File I/O Operations Examples")
    print("=" * 50)

    # Demonstrate file operations
    loaded_airfoils, temp_dir = demonstrate_file_loading()
    output_dir = demonstrate_file_saving()
    batch_airfoils, batch_dir = batch_file_operations()
    validation_results = data_validation_examples()
    integration_with_databases()

    # Create visualizations
    plot_file_operations_results()

    # Cleanup information
    print("\\n" + "=" * 50)
    print("Temporary directories created:")
    print(f"  Input examples: {temp_dir}")
    print(f"  Output examples: {output_dir}")
    print(f"  Batch processing: {batch_dir}")
    print("\\nNote: These directories contain example files for reference")

    print("\\nKey Takeaways:")
    print("1. JAX airfoils support standard file formats (Selig, Lednicer, etc.)")
    print("2. Robust error handling manages corrupted or invalid data")
    print("3. Batch operations enable efficient processing of multiple files")
    print("4. Data validation ensures airfoil quality and consistency")
    print("5. Multiple output formats support different use cases")
    print("6. Integration with databases enables systematic data management")


if __name__ == "__main__":
    main()
