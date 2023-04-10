import sys


def main():

    package = "ICARUS"
    version = "0.0.1"
    if len(sys.argv) >= 2:
        command = sys.argv[1]
    else:
        command = 'install'

    if command == 'install':
        install(package, version)
    elif command == 'uninstall':
        uninstall(package)
    else:
        print("Command not recognized")
        print("Usage: python setup.py [install|uninstall]")
        sys.exit(1)


def install(package, version):
    try:
        from setuptools import setup, find_packages
    except ImportError:
        print("Please install setuptools")
    setup(name=package, version=version, packages=find_packages())

    return


def uninstall(package):
    try:
        import pip
    except ImportError:
        print("Error importing pip")
    pip.main(['uninstall', package, '-y'])


if __name__ == '__main__':
    main()
