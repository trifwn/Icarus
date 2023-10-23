.. ICARUS documentation master file, created by
   sphinx-quickstart on Wed Oct 18 17:05:14 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ICARUS Aerodynamics
======================================

This webpage serves to provide an analysis of the workings and structure of the ICARUS software package, as well as some examples of usage. The package is an open source tool for the design and analysis of aircraft, with emphasis on an abstract, parametric and modular modeling. Most of the source code is currently written in Python. This choice was made deliberately for several reasons:

    1. The Python language enables rabid development and prototyping.
    2. The Python language can be easily read by non-programmers as it resembles natural language more closely than other alternatives.
    3. Python provides a rich ecosystem of computational and numerical open-source packages (e.g. Numpy, Scipy, Scikit-Learn etc.) with high level API’s that when utilized correctly can generate high performance code and implement most algorithms that are needed in engineering tasks.
    4. Object-Oriented programming can make modeling of complex objects (aerial vehicles in our case)  much more concise as it requires partitioning of attributes and functions. This enables the creation of complex workflows in a relatively short time.

The ICARUS package is designed to be used by aircraft designers, engineers and students. All the source code is hosted openly in the following repository: https://github.com/trifwn/Icarus.

Motivation
===========

The project was inspired by an effort to study a model aircraft designed by EUROAVIA Athens for the European Competion “Air Cargo Challenge” (ACC) that failed to fly because of dynamic instability (hence the ICARUS name). The motivation behind developing ICARUS was to provide a versatile and accessible platform for aircraft designers and engineers that expedites research and development. In the area of aerospace most popular software, that can be used to design aircraft fall in one of two categories:

    1. They are problem specific and have a steep learning curve. This category includes most of aerodynamic solvers that are available in academia. Since their purpose is to produce high-fidelity results in a computationally effective manner, they are mostly written in low-level hard to read and even harder to debug languages, that make them difficult to work with. This is especially true, when someone wants to use the code in a not as designed way or incorporate them in a custom workflow.
    2. They are commercial and closed-source. In most cases, they are either not available to the general public or obscured behind paywalls. Typically, these software packages, are addressed to and tailor-made for large corporations. The learning curve of such software can also be cumbersome and prototyping new workflows with them is difficult if not impossible due to their closed-source nature.

On the basis of these ICARUS hopes to provide a high level abstraction between aircraft design and modeling and aerodynamic computations. The way this is achieved, and the choices that enable it, will be discussed further in the following sections.


Code Documentation
==================

Detailed code documentation is available within the ICARUS repository, in the form of code comments, providing users with information on the usage of almost all functions and classes. The documentation is essential for users looking to understand and effectively utilize the capabilities of ICARUS in their aircraft design and analysis process, or contribute to the project.

.. toctree::
   :maxdepth: 4

   modules

Examples
===================
.. toctree::
   :maxdepth: 2

   Examples/examples

Command Line Interface
======================
.. toctree::
   :maxdepth: 2

   CLI/cli
