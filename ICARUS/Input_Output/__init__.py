import os

APPHOME = os.path.abspath("ICARUS")

f2wLoc = os.path.abspath("")
genuLoc = os.path.abspath("")

runOFscript = os.path.join(APPHOME, "Software", "OpenFoam", "runFoam.sh")
setup_of_script = os.path.join(APPHOME, "Software", "OpenFoam", "setupFoam.sh")
logOFscript = os.path.join(APPHOME, "Software", "OpenFoam", "logFoam.sh")
