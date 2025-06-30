"""Genu Surface Class
A class to define lifting surfaces for the GenuVP solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.vehicle import WingSurface


class GenuSurface:
    surf_names: dict[str, int] = {}
    airfoil_names: dict[str, int] = {}

    def __init__(self, surf: WingSurface, idx: int) -> None:
        """Converts a Wing Object to an object that can be used
        for making the input files of GNVP3 and GNVP7

        Args:
            surf (Wing): Wing Object
            idx (int): IND of the surface to be assigned

        Returns:
            dict[str, Any]: Dict with the surface parameters

        """
        if surf.is_symmetric_y:
            N: int = 2 * surf.N - 1
            M: int = surf.M
        else:
            N = surf.N
            M = surf.M

        # We have to make sure that the surface names and airfoil names
        # are unique for each surface and airfoil and also give them names
        # that are valid for GenuVP. This means a total of 12 characters

        # First we check if the surface name is unique
        if surf.name[:8] not in GenuSurface.surf_names.keys():
            GenuSurface.surf_names[surf.name[:8]] = 0
        else:
            GenuSurface.surf_names[surf.name[:8]] += 1
        # Then we check if the airfoil name is unique
        for airfoil in surf.airfoils:
            if airfoil.name[:8] not in GenuSurface.airfoil_names.keys():
                GenuSurface.airfoil_names[airfoil.name[:8]] = 0
            else:
                GenuSurface.airfoil_names[airfoil.name[:8]] += 1

        name: str = f"{GenuSurface.surf_names[surf.name[:8]]}_{surf.name}"

        airfoil_name: str = f"{GenuSurface.airfoil_names[surf.airfoils[0].name[:8]]}_{surf.airfoils[0].name}"

        # Make the names valid for GenuVP and make sure they are unique
        # by adding a number to the end of the name
        name = name[:8]
        airfoil_name = airfoil_name[:8]

        # Now we can start assigning the parameters
        self.type: str = "thin"
        self.lifting: str = "yes"
        self.NB: int = idx
        self.name: str = surf.name
        self.airfoil_name: str = surf.airfoils[0].name
        self.bld_fname: str = f"{name}.bld"
        self.topo_fname: str = f"{name}.topo"
        self.wake_fname: str = f"{name}.wake"
        self.cld_fname: str = f"{airfoil_name}.cld"
        self.move_fname: str = f"{name}.mov"
        self.grid_fname: str = f"{name}.grid"
        self.NNB: int = M
        self.NCWB: int = N
        self.x_0: float = surf.origin[0]
        self.y_0: float = surf.origin[1]
        self.z_0: float = surf.origin[2]
        self.pitch: float = surf.orientation[0] * 0
        self.cone: float = surf.orientation[1] * 0
        self.wngang: float = surf.orientation[2] * 0
        self.x_end: float = surf.origin[0] + surf._xoffset_dist[-1]
        self.y_end: float = surf.origin[1] + surf.span
        self.z_end: float = surf.origin[2] + surf._zoffset_dist[-1]
        self.root_chord: float = surf.chords[0]
        self.tip_chord: float = surf.chords[-1]
        self.offset: float = surf._xoffset_dist[-1]
        self.grid: FloatArray | list[FloatArray] = surf.get_grid()
        self.mean_aerodynamic_chord: float = surf.mean_aerodynamic_chord
