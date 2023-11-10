"""
Setup script for ICARUS
"""
from importlib.metadata import entry_points
import os
import platform
import re
import subprocess
import sys
from typing import Any

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist
from setuptools.command.test import test
from setuptools.command.upload import upload
from wheel.bdist_wheel import bdist_wheel


# from distutils.command.clean import clean
# from setuptools.command.build import build

HOMEDIR = os.getcwd()


def get_package_version() -> str:
    """Get the package version from the __init__ file"""
    __version__: str = re.findall(
        r"""__version__ = ["']+([0-9\.]*)["']+""",
        open("ICARUS/__init__.py", encoding="UTF-8").read(),
    )[0]
    return __version__


def main() -> None:
    """MAIN FUNCTION"""

    package = "ICARUS"
    __version__: str = get_package_version()

    # TODO: Check for intel fortran, opemmpi, mkl
    # TODO: if not installed, install them.

    if len(sys.argv) >= 2:
        command: str = sys.argv[1]
    else:
        command = "install"

    if command == "dist_info":
        setup(cmdclass={"sdist": sdist})
    if command == "editable_wheel":
        setup(cmdclass={"develop": develop})
    if command == "install":
        install(package, __version__)
    elif command == "uninstall":
        uninstall(package)
    else:
        print(f"Command {command} not recognized")


def install(package: str, version: str) -> None:
    """INSTALL THE PACKAGE

    Args:
        package (str): Package Name
        version (str): Version Number
    """

    setup(
        name=package,
        version=version,
        # entry_points = {
        #     'gnvp_wake' 
        # }
    )


def uninstall(package: str) -> None:
    """Uninstall the package

    Args:
        package (str): Package Name
    """
    try:
        import pip
    except ImportError:
        print("Error importing pip")
        return
    pip.main(["uninstall", package, "-y"])


if __name__ == "__main__":
    main()
