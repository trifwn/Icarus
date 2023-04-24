import os

### MOCK CASES ###
# 2D
APPHOME = os.path.abspath("ICARUS")
BASEOPENFOAM = os.path.join(APPHOME, "Database", "Mock", "BaseOF")
BASEFOIL2W = os.path.join(APPHOME, "Database", "Mock", "BaseF2W")
# 3D
BASEGNVP3 = os.path.join(APPHOME, "Database", "Mock", "BaseGNVP3")

### DATABASES ###
DB2D = os.path.join(APPHOME, "Database", "2D")
DB3D = os.path.join(APPHOME, "Database", "3D")
