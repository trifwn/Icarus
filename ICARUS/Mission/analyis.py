class MissionAnalysis:
    def __init__(self, Mission, solver, vehicle) -> None:
        self.mission = Mission
        self.solver = solver
        self.vehicle = vehicle

    def getVehicle(self):
        return self.mission.vehicle

    def setVehicle(self, vehicle):
        self.mission.vehicle = vehicle

    def Analyze(self):
        pass

    def getResults(self):
        pass

    def getFitness(self, fitness):
        return fitness.getFitness(self.getResults())

    def plot(self):
        pass

    def save(self):
        pass
