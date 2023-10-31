"""
============================================
ICARUS Input Output package
============================================

.. toctree: generated/
    :hidden:

    ICARUS.Input_Output.OpenFoam
    ICARUS.Input_Output.Xfoil
    ICARUS.Input_Output.F2Wsection
    ICARUS.Input_Output.GenuVP
    ICARUS.Input_Output.XFLR5

"""
from . import F2Wsection
from . import GenuVP
from . import OpenFoam
from . import XFLR5
from . import Xfoil

_all_ = ["OpenFoam", "Xfoil", "F2Wsection", "GenuVP", "XFLR5"]

import os

APPHOME = os.path.abspath("ICARUS")

f2wLoc = os.path.abspath("")
genuLoc = os.path.abspath("")

runOFscript = os.path.join(APPHOME, "Input_Output", "OpenFoam", "runFoam.sh")
setup_of_script = os.path.join(APPHOME, "Input_Output", "OpenFoam", "setupFoam.sh")
logOFscript = os.path.join(APPHOME, "Input_Output", "OpenFoam", "logFoam.sh")
