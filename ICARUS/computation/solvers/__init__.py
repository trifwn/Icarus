"""============================================
ICARUS Input Output package
============================================

.. toctree: generated/
    :hidden:

    ICARUS.Solvers.OpenFoam
    ICARUS.Solvers.Xfoil
    ICARUS.Solvers.Foil2Wake
    ICARUS.Solvers.GenuVP
    ICARUS.Solvers.XFLR5

"""

from . import XFLR5
from . import Foil2Wake
from . import GenuVP
from . import Icarus_LSPT
from . import OpenFoam
from . import Xfoil

__all__ = ["XFLR5", "Foil2Wake", "GenuVP", "Icarus_LSPT", "OpenFoam", "Xfoil"]

import os

APPHOME = os.path.abspath("ICARUS")

f2wLoc = os.path.abspath("")
genuLoc = os.path.abspath("")

runOFscript = os.path.join(APPHOME, "Solvers", "OpenFoam", "runFoam.sh")
setup_of_script = os.path.join(APPHOME, "Solvers", "OpenFoam", "setupFoam.sh")
logOFscript = os.path.join(APPHOME, "Solvers", "OpenFoam", "logFoam.sh")
