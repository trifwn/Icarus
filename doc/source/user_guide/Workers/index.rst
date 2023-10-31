ICARUS.Workers package
======================

The basic abstraction from which ICARUS derives most of its utility is the Workers module. The naming was chosen as a long term goal of the project is for computations to be distributed to different machines, and therefore the term worker is more appropriate than solver. A Worker is a process that takes some input and computes an outout based on a specified algorithm. Each worker, therefore, is in-terms of our workflow a solver that takes an engineering abstraction, translates it to proper input, performs computations, derives the output, and post-process it to a result we desire.

For instance, when investigating the lift-drag characteristics of a wing: Our internal representation of the wing represents the object we want to study. The worker will need to employ a Solver, for instance GenuVP3, that has been developed to work with specific input and perform computations on it. The task of the worker is to translate our internal representation of a wing in the specific format that the solver expects it, manage the computation from a high level (allocate CPU-cores and memory), track its progress, and at the end collect the results and translate them back to a datatype that is convenient for us.

Recapping the above, in ICARUS all computation is handled by workers which model Solvers that can perform computations, named Analyses, based on the Options we specify! These three abstract concepts are the basis of any workflow we specify and will now be examined separately.


Workers.solver module
-------------------------------------------------------------------------------------------------------------------------------------------------------

.. include a link to the solver module
.. toctree::
    :maxdepth: 1

    solver

A solver is initialized by Calling the following function:

.. literalinclude:: ../../../../ICARUS/Workers/solver.py
    :pyobject: Solver.__init__

We specify the solver with:
    * A name we assign to it. E.g. GenuVP3
    * A type, which serves as a description. E.g. 3D-Vortex-Particle-Method or VPM
    * A fidelity, which describes the accuracy of the results we expect to get from it. As a convention there are three fidelity levels:
        - 0: Representing a zero-cost but extremely inaccurate calculation (e.g. conceptual design)
        - 1: Representing a relatively fast and not completely inaccurate calculation (e.g. statistical models)
        - 2: Representing a relatively slow but somewhat accurate calculation (e.g. 2D Integral Boundary Layer methods for the study of airfoils)
        - 3: Representing a usually costly but accurate calculation (e.g. CFD)



.. highlight:: python

For to define the xfoil solver object we call:

::

    xfoil = Solver(name="xfoil", solver_type="2D-IBLM", fidelity=2, db=db)

After we have our Solver Object we can add analyses to it.

Workers.analysis module
--------------------------------------------------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 1

    analysis

An analysis is the bread-and-butter of computation it has to define:
    * How we communicate with our solver
    * What data we need to pass to it
    * What options we need to specify for it to run
    * How we must pre-process and  post-process the data

We initialize an analysis by calling the following:

.. literalinclude:: ../../../../ICARUS/Workers/analysis.py
    :pyobject: Analysis.__init__

To define an analysis we need to specify:
    * The name of the solver it corresponds to.
    * The name of the analysis.
    * The function that is run to perform the analysis.
    * The options that should be provided for the above function.
    * The solver options we can tune. (Internal Solver Variables that we can specify or leave to their default value)
    * An unhook function that is run once the solver has run and produced the results need.

.. highlight:: python

For example we can define the following analysis:

::

    aseq_multiple_reynolds_serial: Analysis = Analysis(
        solver_name="xfoil",
        analysis_name="Aseq for Multiple Reynolds Sequentially",
        run_function=multiple_reynolds_serial,
        options=options,
        solver_options=solver_options,
        unhook=None,
    )

As we can see this analysis has the following properties:
    * It is associated with the xfoil solver.
    * It is called "Aseq for Multiple Reynolds Sequentially". This means that this analysis will run the xfoil solver's aseq analysis in a sequential manner for multiple reynolds numbers.
    * When executed it calls the multiple_reynolds_serial function.
    * It requires the options specified in the options variable.
    * It requires the solver_options specified in the solver_options variable.
    * It does not have an unhook function (==None).


.. highlight:: python

The options we passed on the above analysis are defined in the following section are:

::

    options: dict[str, tuple[str, Any]] = {
        "db": (
            "Database to save results",
            DB,
        ),
        "airfoil": (
            "Airfoil to run",
            Airfoil,
        ),
        "reynolds": (
            "List of Reynolds numbers to run",
            list[float],
        ),
        "mach": (
            "Mach number",
            float,
        ),
        "min_aoa": (
            "Minimum angle of attack",
            float,
        ),
        "max_aoa": (
            "Maximum angle of attack",
            float,
        ),
        "aoa_step": (
            "Step between each angle of attack",
            float,
        ),
    }


.. highlight:: python

In the same manner the solver_options are defined in the following section:

::

    solver_options: dict[str, tuple[Any, str, Any]] = {
        "max_iter": (
            100,
            "Maximum number of iterations",
            int,
        ),
        "Ncrit": (
            1e-3,
            "Timestep between each iteration",
            float,
        ),
        "xtr": (
            (0.1, 0.1),
            "Transition points: Lower and upper",
            tuple[float, float],
        ),
        "print": (
            False,
            "Print xfoil output",
            bool,
        ),
    }


.. highlight:: python

Finnaly the function that is called when the analysis is run is defined in the following section (This function is defined in the Input_Output module):

::

    def multiple_reynolds_serial(
        db: DB,
        airfoil: Airfoil,
        reynolds: list[float],
        mach: float,
        min_aoa: float,
        max_aoa: float,
        aoa_step: float,
        solver_options: dict[str, Any],
    ) -> None:
    """
    Function that take the above options as input arguments and runs the analysis.
    """


Workers.options module
-------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 1

    options


.. literalinclude:: ../../../../ICARUS/Workers/options.py
    :pyobject: Option.__init__

In the above section we defined the options that are required for the analysis to run. We passed them in the options variable as a dictionary. These options will be converted to Option objects. Every Option therefore is defined by its:
    * name: The name of the option. This name will be used as a keyword argument in the associated analysis run and unhook functions, so extra caution must be given when naming options.
    * value: The value of the option.
    * description: A description of the option in text.
    * option_type: The type of the option. (e.g. int, float, list, etc.)


.. highlight:: python

For example the following option is defined:

::

        Option(name='reynolds',
            value=[1e6,2e6,3e6,4e6,5e6,6e6],
            description='List of Reynolds numbers to run',
            option_type=list[float],
        )


Implementation
----------------------------------------------------------------------------------------

The implementation of the above concepts is done in the solvers modules where we define all solvers that are available to ICARUS. To read more about the implementation of the solvers please refer to the Solvers section of the documentation.
