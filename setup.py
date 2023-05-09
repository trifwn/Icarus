"""
Setup script for ICARUS
"""
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
    __version__ = re.findall(
        r"""__version__ = ["']+([0-9\.]*)["']+""",
        open("ICARUS/__init__.py", encoding="UTF-8").read(),
    )[0]
    return __version__


class Repository:
    """Class object for a repository"""

    def __init__(self, name, url, MakeType) -> None:
        self.url = url
        self.name = name
        self.type = MakeType
        self.repo_dir: str

    def clone(self) -> None:
        """Clone the Repository"""
        self.repo_dir = os.path.join(HOMEDIR, "3d_Party", self.name)
        clone_cmd = f"git clone {self.url} {self.repo_dir}"
        try:
            subprocess.call(clone_cmd.split())
        except Exception:
            print(
                f"Failed to clone {self.name} repository. Please make sure git is installed and try again.",
            )


class BuildExtension(Extension):
    """Class object for building an extension using Make"""

    def __init__(
        self,
        name,
        make_list_dir,
        makeType,
        configire_commands,
        **kwargs,
    ) -> None:
        super().__init__(name, sources=[], **kwargs)
        self.make_lists_dir = os.path.abspath(make_list_dir)
        self.type = makeType
        self.configire_commands = configire_commands


class MakeBuild(build_ext):
    """Class object Build an extension using Make"""

    def build_extensions(self) -> None:
        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Release"

            if ext.type == "CMake":
                # Ensure that CMake is present and working
                try:
                    _ = subprocess.check_output(["cmake", "--version"])
                except OSError as exc:
                    raise RuntimeError("Cannot find CMake executable") from exc
                cmake_args = [
                    "-DCMAKE_BUILD_TYPE=%s" % cfg,
                    # Ask CMake to place the resulting library in the directory containing the extension
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(),
                        extdir,
                    ),
                    # Other intermediate static libraries are placed in a temporary build directory instead
                    "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(),
                        self.build_temp,
                    ),
                    # Hint CMake to use the same Python executable that is launching
                    # the build, prevents possible mismatching if
                    # multiple versions of Python are installed
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                ]
                # We can handle some platform-specific settings at our discretion
                if platform.system() == "Windows":
                    plat = "x64" if platform.architecture()[0] == "64bit" else "Win32"
                    cmake_args += [
                        # These options are likely to be needed under Windows
                        "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
                        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(
                            cfg.upper(),
                            extdir,
                        ),
                    ]
                    # Assuming that Visual Studio and MinGW are supported compilers
                    if self.compiler.compiler_type == "msvc":
                        cmake_args += [
                            "-DCMAKE_GENERATOR_PLATFORM=%s" % plat,
                        ]
                    else:
                        cmake_args += [
                            "-DCMAKE_CXX_COMPILER=g++",
                            "-DCMAKE_C_COMPILER=gcc",
                            "-DCMAKE_FORTRAN_COMPILER=gfortran",
                            "-G",
                            "MinGW Makefiles",
                        ]

                if not os.path.exists(self.build_temp):
                    os.makedirs(self.build_temp)

                # Config and build the extension
                subprocess.check_call(
                    ["cmake", ext.make_lists_dir] + cmake_args,
                    cwd=self.build_temp,
                )
                subprocess.check_call(
                    ["cmake", "--build", ".", "--config", cfg],
                    cwd=self.build_temp,
                )

            elif ext.type == "make":
                # Run the MAKE command
                if platform.system() == "Windows":
                    make_cmd = "mingw32-make.exe"
                else:
                    make_cmd = "make"

                if not os.path.exists(self.build_temp):
                    os.makedirs(self.build_temp)
                print(ext.make_lists_dir)
                subprocess.check_call([make_cmd, "gnu"], cwd=ext.make_lists_dir)

            elif ext.type == "pip":
                pass
            else:
                print(f"Dont know how to make type {ext.type}")


repos: dict[str, dict[str, Any]] = {
    "CGNS": {
        "url": "https://github.com/CGNS/CGNS.git",
        "configure_commands": [],
        "type": "CMake",
    },
    "structAirfoilMesher": {
        "url": "https://gitlab.com/before_may/structAirfoilMesher.git",
        "configure_commands": [],
        "type": "make",
    },
}


def main() -> None:
    """MAIN FUNCTION"""

    __version__: str = get_package_version()

    # Should Check for intel fortran, opemmpi, mlk

    # Command line flags forwarded to CMake
    if len(sys.argv) >= 2:
        command: str = sys.argv[1]
    else:
        command = "install"

    package = "ICARUS"

    if command == "install":
        print(f"Installing {package} version {__version__}")
        install(package, __version__)
    elif command == "uninstall":
        print(f"Uninstalling {package}")
        uninstall(package)
    elif command == "egg_info":
        print(f"Generating metadata for {package} version {__version__}")
        setup(cmdclass={"egg_info": egg_info})
    elif command == "dist_info":
        print(
            f"Generating source distribution metadata for {package} version {__version__}",
        )
        setup(cmdclass={"sdist": sdist})
    elif command == "bdist_wheel":
        print(f"Generating wheel distribution for {package} version {__version__}")
        setup(cmdclass={"bdist_wheel": bdist_wheel})  # type: ignore
    # elif command == "clean":
    #     print("Cleaning up...")
    #     setup(cmdclass={"clean": clean})
    # elif command == "build":
    #     print("Building...")
    #     setup(cmdclass={"build": build})
    elif command == "develop":
        print("Installing package in development mode...")
        setup(cmdclass={"develop": develop})
    elif command == "test":
        print("Running tests...")
        setup(cmdclass={"test": test})
    elif command == "upload":
        print("Uploading package...")
        setup(cmdclass={"upload": upload})
    else:
        print(f"Command {command} not recognized")
        print(
            "Usage: python setup.py [install|uninstall|egg_info|dist_info|bdist_wheel|clean|build|develop|test|upload]",
        )
        sys.exit(1)


def install(package: str, version: str) -> None:
    """INSTALL THE PACKAGE

    Args:
        package (str): Package Name
        version (str): Version Number
    """
    # ext_modules = []
    # for repo in repos.keys():
    #     repo = repository(repo, repos[repo]['url'], repos[repo]['type'])
    #     # repo.clone()
    #     ext_modules.append(BuildExtension(name = repo.name, make_list_dir = repo.repo_dir, makeType= repo.type ))

    setup(
        name=package,
        version=version,
        # ext_modules= ext_modules,
        # cmdclass={'build_ext': MakeBuild},
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
