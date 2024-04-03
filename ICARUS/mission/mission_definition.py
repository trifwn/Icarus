from typing import Any

from ICARUS.core.struct import Struct
from ICARUS.mission.segment.mission_segment import MissionSegment


class Mission:
    def __init__(
        self,
        segments: list[MissionSegment],
        fitness: Any,
        constraints: Any,
    ) -> None:
        self.missionSegments: Struct = Struct()
        for segment in segments:
            self.addSegment(segment)
        pass
        self.fitness = fitness
        self.constrains = constraints

    def addSegment(self, segment: MissionSegment) -> None:
        self.missionSegments[segment.name] = segment
        # self.missionSegments.sort(key=lambda x: x.startingTime) ## NOT IMPLEMENTED
        pass
