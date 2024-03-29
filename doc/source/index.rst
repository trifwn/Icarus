.. module:: ICARUS

*********************
ICARUS aerodynamics
*********************

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


.. grid:: 2

   .. grid-item-card::
      :img-top: _static/pictures/documentation.jpg

      Code Documentation
      ^^^^^^^^^^^^^^^^^^^^^^^

      Documentation of the ICARUS codebase.

      +++

      .. button-ref:: documentation
         :expand:
         :color: secondary
         :click-parent:

         To the documentation reference

   .. grid-item-card::
      :img-top: _static/pictures/cli.jpg

      Command Line Interface
      ^^^^^^^^^^^^^^^^^^^^^^^

      Guide to using the ICARUS command line interface.

      +++

      .. button-ref:: cli
         :expand:
         :color: secondary
         :click-parent:

         To the Command Line Interface guide


   .. grid-item-card::
      :img-top: _static/pictures/user_guide.jpg

      Example Usage
      ^^^^^^^^^^^^^^^^^^^^^^^

      Some examples of using ICARUS in the design and analysis of aircraft.

      +++

      .. button-ref:: user_guide
         :expand:
         :color: secondary
         :click-parent:

         To the example usage guide

   .. grid-item-card::
      :img-top: _static/pictures/tutorials.jpg

      Tutorials
      ^^^^^^^^^^^^^^^^^^^^^^^

      Code Notebooks that demonstrate the use of ICARUS.

      +++

      .. button-ref:: tutorials
         :expand:
         :color: secondary
         :click-parent:

         To the Tutorials page

.. toctree::
   :maxdepth: 1
   :hidden:

   Code Documentation <reference/index>
   Command Line Interface <cli/index>
   User Guide <user_guide/index>
   Tutorials <tutorials/index>
