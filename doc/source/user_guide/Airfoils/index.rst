ICARUS.airfoils package
=======================

The airfoils module serves to module 2D airfoil geometries. This library is an extension of an already available module that can be found in https://pypi.org/project/airfoils/. An airfoil is defined by specifying the geometry of the upper (suction) and lower (pressure) side, the number of points  that was given, and a name for the airfoil.In the module we also provide functions that:

    • Produce 4 and 5 digit NACA geometries.
    • Export the airfoil to different formats (selig and more)
    • Interpolate points that are not on the geometry grid
    • Compute the camber line
    • Plot the airfoil
    • Flap the airfoil
    • Morph an airfoil by interpolating 2 airfoils and a relative position

Part of the airfoils module is also the AirfoilPolars class. The function of this object is to take computed 2D polars for different Reynolds and Mach numbers and interpolate in-between values that we might need.


Airfoil module
------------------------------------

.. :ref:`ICARUS.airfoils.airfoil`

The basic module for the airfoils package is the airfoil module. This module contains the Airfoil class that is used to create an airfoil object. The airfoil class is a superset of the airfoils pypi package (https://pypi.org/project/airfoils/) that adds usefull functionality. The goal of ICARUS is to eventually deprecate this dependency and incorporate all the functionality of the airfoils package in the ICARUS.airfoils.airfoil module.

To create an airfoil object we need to import the module and then call the Airfoil class. The following example creates an airfoil object for the NACA 2412 airfoil using the built in naca class-method. The naca method currently supports 4 and 5 digit NACA airfoils.

.. code-block:: python

    from ICARUS.airfoils.airfoil import Airfoil

    airfoil = Airfoil.naca("NACA2412")

We can see the geometry of the airfoil by calling the plot function.

.. code-block:: python

    airfoil.plot()

.. image:: ../../_static/pictures/naca2412.png

In general we can create an airfoil by specifying the upper and lower side of the airfoil. The initializer for an airfoil object is:


.. literalinclude:: ../../../../ICARUS/airfoils/airfoil.py
    :pyobject: Airfoil.__init__

To create an airfoil if we have a specific geometry that includes the upper and lower side we can do the following:

.. code-block:: python

    upper_side = [
        [0., 0.],
         ... ,
        [1., 0.].
    ] # list of coordinates for the upper side can also be a numpy array
    lower_side = [
        [0., 0.],
         ... ,
        [1., 0.].
    ] # list of coordinates for the lower side can also be a numpy array
    name = "my_airfoil"
    n_points = 100
    my_airfoil = Airfoil(upper_side, lower_side, name, n_points)


We can also create a flapped version of our airfoil by calling the airfoil.flap_airfoil function:

.. code-block:: python

    airfoil_flapped = airfoil.flap_airfoil(
        flap_hinge= 0.7,
        chord_extension= 1.3,
        flap_angle= 20,
    )
    airfoil_flapped.plot()

.. image:: ../../_static/pictures/naca2412_flapped.png

Airfoil Polars module
--------------------------------------

.. :ref:`ICARUS.airfoils.airfoil_polars`

The airfoil module also contains the Polars class that is used to handle the polars we calculate for a specific airfoil. The basic functionality of the class is to interpolate polars for different reynolds and mach numbers from computed states.
