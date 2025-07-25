#!/usr/bin/env python3
"""
Simple runner for the Fibonacci Demo

This script provides an easy way to run the Fibonacci simulation demo
with predefined configurations.
"""

import subprocess
import sys
from pathlib import Path


def run_demo(config_name: str) -> None:
    """Run the demo with a specific configuration."""
    demo_script = Path(__file__).parent / "fibonacci_demo.py"

    configs = {
        "quick": ["--numbers"] + list(map(str, range(10, 16))) + ["--delay", "0.05"],
        "standard": ["--numbers"] + list(map(str, range(8, 23))) + ["--delay", "0.05"],
        "performance": ["--numbers"]
        + list(map(str, range(5, 31)))
        + ["--delay", "0.05"],
        "sequential": ["--numbers"]
        + list(map(str, range(10, 21)))
        + ["--mode", "sequential"],
        "async": ["--numbers"] + list(map(str, range(10, 21))) + ["--mode", "async"],
        "threading": ["--numbers"]
        + list(map(str, range(10, 21)))
        + ["--mode", "threading"],
        "multiprocessing": ["--numbers"]
        + list(map(str, range(10, 21)))
        + ["--mode", "multiprocessing"],
    }

    if config_name not in configs:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available configurations: {', '.join(configs.keys())}")
        sys.exit(1)

    args = configs[config_name]
    cmd = [sys.executable, str(demo_script)] + args

    print(f"🚀 Running Fibonacci Demo - {config_name.upper()} configuration")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
        sys.exit(1)


def main() -> None:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python run_fibonacci_demo.py <config_name>")
        print("\nAvailable configurations:")
        print("  quick      - Quick demo with small numbers (F10-F15)")
        print("  standard   - Standard demo (F8-F22)")
        print("  performance- Performance test with larger range (F5-F30)")
        print("  sequential - Test sequential execution only")
        print("  async      - Test async execution only")
        print("  threading  - Test threading execution only")
        print("\nExample: python run_fibonacci_demo.py standard")
        # sys.exit(1)
        config_name = "standard"
    else:
        config_name = sys.argv[1].lower()
    run_demo(config_name)


if __name__ == "__main__":
    main()
