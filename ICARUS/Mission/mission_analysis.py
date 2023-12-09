from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Mission.mission_definition import Mission
from ICARUS.Vehicle.plane import Airplane


class MissionAnalysis:
    def __init__(self, mission: Mission, solver: Solver, vehicle: Airplane) -> None:
        self.mission: Mission = mission
        self.solver: Solver = solver
        self.vehicle: Airplane = vehicle

    def getVehicle(self) -> Airplane:
        return self.vehicle

    def setVehicle(self, vehicle: Airplane) -> None:
        self.vehicle = vehicle

    def Analyze(self) -> None:
        pass

    def getResults(self) -> None:
        pass

    # def getFitness(self, fitness):
    #     return fitness.getFitness(self.getResults())

    def plot(self) -> None:
        pass

    def save(self) -> None:
        pass
