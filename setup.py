from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
import subprocess
import platform
import sys

import os 
import re

HOMEDIR = os.getcwd()
__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('ICARUS/__init__.py').read(),
)[0]

options = {k: 'OFF' for k in ['--opt', '--debug', '--cuda']}

for flag in options.keys():
    if flag in sys.argv:
        options[flag] = 'ON'
        sys.argv.remove(flag)

# Command line flags forwarded to CMake
cmake_cmd_args = []
for f in sys.argv:
    if f.startswith('-D'):
        cmake_cmd_args.append(f)
        sys.argv.remove(f)


class repository():
    def __init__(self, name, url, MakeType):
        self.url = url
        self.name = name
        self.type = MakeType

    def clone(self):
        # Clone the repository
        self.repoDir = os.path.join(HOMEDIR,'3d_Party', self.name)
        clone_cmd = f'git clone {self.url} {self.repoDir}'
        try:
            subprocess.call(clone_cmd.split())
        except:
            print(f'Failed to clone {self.name} repository. Please make sure git is installed and try again.')


class MakeBuild(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = 'Debug' if options['--debug'] == 'ON' else 'Release'

            if ext.type == "CMake":
                cmake_args = [
                    '-DCMAKE_BUILD_TYPE=%s' % cfg,
                    # Ask CMake to place the resulting library in the directory
                    # containing the extension
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                    # Other intermediate static libraries are placed in a
                    # temporary build directory instead
                    '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), self.build_temp),
                    # Hint CMake to use the same Python executable that
                    # is launching the build, prevents possible mismatching if
                    # multiple versions of Python are installed
                    '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                    # Add other project-specific CMake arguments if needed
                    # ...
                ]

                # We can handle some platform-specific settings at our discretion
                if platform.system() == 'Windows':
                    plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
                    cmake_args += [
                        # These options are likely to be needed under Windows
                        '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                        '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                    ]
                    # Assuming that Visual Studio and MinGW are supported compilers
                    if self.compiler.compiler_type == 'msvc':
                        cmake_args += [
                            '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                            ]
                    else:
                        cmake_args += [
                            '-G', 'MinGW Makefiles',
                        ]

                cmake_args += cmake_cmd_args

                print(cmake_args)

                if not os.path.exists(self.build_temp):
                    os.makedirs(self.build_temp)

                # Config and build the extension
                subprocess.check_call(['cmake', ext.make_lists_dir] + cmake_args,
                                    cwd=self.build_temp)
                subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
                                    cwd=self.build_temp)
            elif ext.type == "make":
                # Run the MAKE command
                if platform.system() == 'Windows':
                    make_cmd = f'mingw32-make.exe'
                else:
                    make_cmd = f'make'

                if not os.path.exists(self.build_temp):
                    os.makedirs(self.build_temp)
                print(ext.make_lists_dir)
                subprocess.check_call([make_cmd, 'gnu'], cwd=ext.make_lists_dir)
            else:
                print(f"Dont know how to make type {ext.type}")

class MakeExtension(Extension):
    def __init__(self, name, make_list_dir, makeType, **kwargs):
        super().__init__(name, sources=[], **kwargs)
        self.make_lists_dir = os.path.abspath(make_list_dir)
        self.type = makeType

repos = {
    'structAirfoilMesher': {
            'url': 'https://gitlab.com/before_may/structAirfoilMesher.git',
            'type': "make"
            },
}

ext_modules = []
for repo in repos.keys():
    repo = repository(repo, repos[repo]['url'], repos[repo]['type'])
    repo.clone()
    ext_modules.append(MakeExtension(name = repo.name, make_list_dir = repo.repoDir, makeType= repo.type ))

setup(
    version= __version__,
    # ext_modules= ext_modules,
    # cmdclass={'build_ext': MakeBuild},
)