from ICARUS.computation.solvers.solver import Solver
from ICARUS.mission.mission import Mission
from ICARUS.vehicle.airplane import Airplane


class MissionAnalysis:
    def __init__(self, mission: Mission, solver: Solver, vehicle: Airplane) -> None:
        self.mission: Mission = mission
        self.solver: Solver = solver
        self.vehicle: Airplane = vehicle

    def get_vehicle(self) -> Airplane:
        return self.vehicle

    def set_vehicle(self, vehicle: Airplane) -> None:
        self.vehicle = vehicle

    def analyze(self) -> None:
        pass

    def process(self) -> None:
        pass

    def plot(self) -> None:
        pass

    def save(self) -> None:
        pass
