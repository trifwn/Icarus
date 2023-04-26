import os

### MOCK CASES ###
# 2D
APPHOME = os.path.abspath(".")

BASEOPENFOAM = os.path.join(APPHOME, "Data", "Mock", "BaseOF")
BASEFOIL2W = os.path.join(APPHOME, "Data", "Mock", "BaseF2W")
# 3D
BASEGNVP3 = os.path.join(APPHOME, "Data", "Mock", "BaseGNVP3")

### DATABASES ###
DB2D = os.path.join(APPHOME, "Data", "2D")
DB3D = os.path.join(APPHOME, "Data", "3D")
ANALYSESDB = os.path.join(APPHOME, "Data", "Analyses")
XFLRDB = os.path.join(APPHOME, "Data", "XFLR5")