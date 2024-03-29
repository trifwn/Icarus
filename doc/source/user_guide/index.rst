.. _user_guide:

User Guide
========================================

The ICARUS package is comprised of different modules that interact with each other but are mostly self-contained in terms of functionality. The modules that comprise ICARUS will be discussed individually in the links below. During the explanation of each of the modules effort will be given to provide the reader some usage examples. The order of the modules has been picked to give the reader a logical progression of the modules. If you just stumbled upon ICARUS it is recommended to read them in order

In this user guide we will cover the usage of the modules in an order that makes sense for a new user. The order of the modules is as follows:

* `Computation`
    - `computation.solvers.solver`
    - `computation.analysis`
    - `computation.options`

* `airfoils`
    - `airfoils.airfoil`
    - `airfoils.airfoil_polars`

* `solvers`
    - `solvers.aifoil`
    - `solvers.airplane`

* `vehicle`
    - `vehicle.strip`
    - `vehicle.wing_segment`
    - `vehicle.merged_wing`


.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

    computation                 <computation/index>
    computation.solver          <computation/solver>
    computation.analysis        <computation/analysis>
    computation.options         <computation/options>
    airfoils                <airfoils/index>
    airfoils.airfoil        <airfoils/airfoil>
    airfoils.airfoil_polars <airfoils/airfoil_polars>
    solvers                 <solvers/index>
    solvers.aifoil          <solvers/airfoil_solvers>
    solvers.airplane        <solvers/airplane_solvers>
    vehicle                 <vehicle/index>
    vehicle.strip           <vehicle/strip>
    vehicle.wing_segment    <vehicle/wing_segment>
    vehicle.merged_wing     <vehicle/merged_wing>
