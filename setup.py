"""
Setup script for ICARUS
"""

import os
import re
import sys

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.sdist import sdist

# import platform
# from typing import Any
# import subprocess
# from setuptools import Extension
# from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info

# from setuptools.command.upload import upload
# from setuptools.command.test import test
# from wheel.bdist_wheel import bdist_wheel


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


from setuptools import setup, find_packages
import subprocess

setup(
    name="ICARUS",
    version=get_package_version(),
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "icarus=icarus_cli:main",
        ],
    },
)


def uninstall(package: str) -> None:
    """Uninstall the package

    Args:
        package (str): Package Name
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "-y"])
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling package: {e}")

