class MissionAnalysis:
    def __init__(self, Mission, solver, vehicle) -> None:
        self.mission = Mission
        self.solver = solver
        self.vehicle = vehicle

    def getVehicle(self):
        return self.mission.vehicle

    def setVehicle(self, vehicle) -> None:
        self.mission.vehicle = vehicle

    def Analyze(self) -> None:
        pass

    def getResults(self) -> None:
        pass

    def getFitness(self, fitness):
        return fitness.getFitness(self.getResults())

    def plot(self) -> None:
        pass

    def save(self) -> None:
        pass
