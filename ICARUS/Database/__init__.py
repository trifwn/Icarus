import os

# MOCK CASES ###
# 2D
APPHOME: str = os.path.dirname(os.path.realpath(__file__))
APPHOME = os.path.abspath(os.path.join(APPHOME, os.pardir))
APPHOME = os.path.abspath(os.path.join(APPHOME, os.pardir))

BASEOPENFOAM: str = os.path.join(APPHOME, "Data", "Mock", "BaseOF")
BASEFOIL2W: str = os.path.join(APPHOME, "Data", "Mock", "BaseF2W")
# 3D
BASEGNVP3: str = os.path.join(APPHOME, "Data", "Mock", "BaseGNVP3")

# DATABASES ###
DB2D: str = os.path.join(APPHOME, "Data", "2D")
DB3D: str = os.path.join(APPHOME, "Data", "3D")
ANALYSESDB: str = os.path.join(APPHOME, "Data", "Analyses")
XFLRDB: str = os.path.join(APPHOME, "Data", "XFLR5")
