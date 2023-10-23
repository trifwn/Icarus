ICARUS.Solvers package
======================

Currently there exist 6 solvers in ICARUS. The ICARUS module is structured so that the addition of new Solvers is as fast and cut-and-dry as possible so that in the future more and more capability is added.

The Solvers that are incorporated are divided into two main categories (subpackages): Airfoil and Airplane solvers. Each solver is described by a solver-file that defines the attributes we described above.


Airfoil Solvers
-----------------------


.. toctree::
   :maxdepth: 2

   Solvers/ICARUS.Solvers.Airfoil


The common options for an airfoil analysis are:

    1. db: The Database to save the results
    2. airfoil: The airfoil we want to run
    3. reynolds: A list of reynolds numbers to run
    4. mach: The mach number to account for compressibility effects
    5. angles: A list of angles to run simulations for


Airplane Solvers
--------------------------

.. toctree::
   :maxdepth: 2

   Solvers/ICARUS.Solvers.Airplane



The common options for an airplane analysis are:

    • db: The Database to save the results
    • plane: The plane object to simulate
    • solver2D: The solver-name that produced the 2D polars to integrate
    • u_freestream: The free-stream wind speed magnitude
    • environment: The environment to be used for calculations
