from typing import Any

from ICARUS.mission.segment.mission_segment import MissionSegment


class Mission:
    def __init__(
        self,
        segments: list[MissionSegment],
        fitness: Any,
        constraints: Any,
    ) -> None:
        self.segments: dict[str, MissionSegment] = {}
        for segment in segments:
            self.add_segment(segment)
        self.fitness = fitness
        self.constrains = constraints

    def add_segment(self, segment: MissionSegment) -> None:
        self.segments[segment.name] = segment
