Airplane Solvers
==============================

The airplane solvers are a set of solvers that are used to solve the flow around an airplane. The solvers currently available are:

* `GenuVP3`: A solver for the Vortex Particle Method (VPM) that uses a 3D panel method to calculate the influence of the wake on the airplane. The software was developed by Spyros Voutsinas at the University of Athens and is available upon request.
* `GenuVP7`:  The next version of GenuVP3 which supports parrallelization using MPI. The software was developed by Spyros Voutsinas at the University of Athens and is available upon request.
* `LSPT`: A 3D panel method that uses a vortex lattice method to solve the potential flow around wings developed specifically for ICARUS


GenuVP3
--------------------------------------

.. include:: ../../../ICARUS/Solvers/Airplane/gnvp3.py
   :literal:


GenuVP7
--------------------------------------

.. include:: ../../../ICARUS/Solvers/Airplane/gnvp7.py
   :literal:


LSPT
--------------------------------------

.. include:: ../../../ICARUS/Solvers/Airplane/lspt.py
   :literal:
