Airfoil solvers
=======================================

The airfoil solvers are a set of tools for solving the flow around airfoils. The solvers currently available are:

* `OpenFoam`: A solver for solving the flow around airfoils using the OpenFoam CFD package. The solver is based on the `simpleFoam` solver and uses the `kOmegaSST` turbulence model. The mesh is obtained using the `structAirfoilMesher` tool developed by Konstantinos Diakakis at https://gitlab.com/before_may/structAirfoilMesher.
* `Xfoil`: An IBLM solver for solving the 2D flow around subsonic airfoils. The software was developed at MIT by proffessor Mark Drela and is available at http://web.mit.edu/drela/Public/web/xfoil/. In ICARUS a wrapped version of the software is used, which is available at https://github.com/DARcorporation/xfoil-python
* `Foil2Wake`: A 2D IBLM solver for solving the flow around airfoils. The software was developed at NTUA and is available upon request.

For each of the solvers we will examine their configuration files that are found in the ICARUS/solvers/Airfoil folders

OpenFoam
--------------------------------------

.. include:: ../../../../ICARUS/solvers/Airfoil/open_foam.py
   :literal:


Xfoil
--------------------------------------

.. include:: ../../../../ICARUS/solvers/Airfoil/xfoil.py
   :literal:


Foil2Wake
--------------------------------------

.. include:: ../../../../ICARUS/solvers/Airfoil/f2w_section.py
   :literal:
