from ICARUS.Core.struct import Struct


class Mission:
    def __init__(self, segments, fitness, constraints) -> None:
        self.missionSegments = Struct()
        for segment in segments:
            self.addSegment(segment)
        pass
        self.fitness = fitness
        self.constrains = constraints

    def addSegment(self, segment):
        self.missionSegments[segment.name] = segment
        # self.missionSegments.sort(key=lambda x: x.startingTime) ## NOT IMPLEMENTED
        pass
