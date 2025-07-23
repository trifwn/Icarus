# JAX Airfoil API Reference

## Overview

The JAX Airfoil implementation provides a fully JAX-compatible airfoil class with JIT compilation and automatic differentiation support. This document provides comprehensive API documentation for all public methods and properties.

## JaxAirfoil Class

### Constructor

```python
JaxAirfoil(
    coordinates: Optional[Union[Array, np.ndarray]] = None,
    name: str = "JaxAirfoil",
    buffer_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

**Description**: Initialize a JaxAirfoil instance with the given coordinates.

**Parameters**:
- `coordinates` (Optional[Union[Array, np.ndarray]]): Airfoil coordinates in selig format (2, n_points) where first row is x, second is y. If None, creates an empty airfoil.
- `name` (str): Name of the airfoil. Default: "JaxAirfoil"
- `buffer_size` (Optional[int]): Optional buffer size to use (auto-determined if None)
- `metadata` (Optional[Dict[str, Any]]): Optional additional metadata

**Raises**:
- `ValueError`: If coordinates have invalid shape or contain invalid values

**Example**:
```python
# Create from coordinates
coords = jnp.array([[1.0, 0.5, 0.0, 0.5, 1.0],
                   [0.0, 0.05, 0.08, -0.05, 0.0]])
airfoil = JaxAirfoil(coords, name="CustomAirfoil")

# Create empty airfoil
empty_airfoil = JaxAirfoil()
```

### Class Methods

#### NACA Generation Methods

##### `naca4(digits, n_points=200, buffer_size=None, metadata=None)`

```python
@classmethod
def naca4(
    cls,
    digits: str,
    n_points: int = 200,
    buffer_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> "JaxAirfoil"
```

**Description**: Create a NACA 4-digit airfoil using JAX operations.

**Parameters**:
- `digits` (str): NACA 4-digit designation (e.g., "2412")
- `n_points` (int): Number of points for each surface. Default: 200
- `buffer_size` (Optional[int]): Optional buffer size (auto-determined if None)
- `metadata` (Optional[Dict[str, Any]]): Optional additional metadata

**Returns**: JaxAirfoil instance with NACA 4-digit airfoil

**Raises**:
- `ValueError`: If digits are invalid (not 4 digits, contains non-numeric characters, etc.)

**Example**:
```python
# Create NACA 2412 airfoil
naca2412 = JaxAirfoil.naca4("2412", n_points=150)

# With custom buffer size
naca0012 = JaxAirfoil.naca4("0012", n_points=100, buffer_size=256)
```

##### `naca5(digits, n_points=200, buffer_size=None, metadata=None)`

```python
@classmethod
def naca5(
    cls,
    digits: str,
    n_points: int = 200,
    buffer_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> "JaxAirfoil"
```

**Description**: Create a NACA 5-digit airfoil using JAX operations.

**Parameters**:
- `digits` (str): NACA 5-digit designation (e.g., "23012")
- `n_points` (int): Number of points for each surface. Default: 200
- `buffer_size` (Optional[int]): Optional buffer size (auto-determined if None)
- `metadata` (Optional[Dict[str, Any]]): Optional additional metadata

**Returns**: JaxAirfoil instance with NACA 5-digit airfoil

**Raises**:
- `ValueError`: If digits are invalid

**Example**:
```python
# Create NACA 23012 airfoil
naca23012 = JaxAirfoil.naca5("23012", n_points=200)
```

##### `naca(designation, n_points=200, buffer_size=None, metadata=None)`

```python
@classmethod
def naca(
    cls,
    designation: str,
    n_points: int = 200,
    buffer_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> "JaxAirfoil"
```

**Description**: Create a NACA airfoil from a designation string. Automatically detects whether the designation is for a 4-digit or 5-digit NACA airfoil.

**Parameters**:
- `designation` (str): NACA designation string (e.g., "2412", "23012")
- `n_points` (int): Number of points for each surface. Default: 200
- `buffer_size` (Optional[int]): Optional buffer size (auto-determined if None)
- `metadata` (Optional[Dict[str, Any]]): Optional additional metadata

**Returns**: JaxAirfoil instance with the specified NACA airfoil

**Raises**:
- `ValueError`: If designation is invalid or unsupported

**Example**:
```python
# Auto-detect NACA type
naca2412 = JaxAirfoil.naca("2412")      # 4-digit
naca23012 = JaxAirfoil.naca("23012")    # 5-digit
naca_with_prefix = JaxAirfoil.naca("NACA2412")  # Handles prefix
```

#### Construction Methods

##### `from_upper_lower(upper, lower, name="JaxAirfoil", buffer_size=None, metadata=None)`

```python
@classmethod
def from_upper_lower(
    cls,
    upper: Union[Array, np.ndarray],
    lower: Union[Array, np.ndarray],
    name: str = "JaxAirfoil",
    buffer_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> "JaxAirfoil"
```

**Description**: Create a JaxAirfoil from separate upper and lower surface coordinates.

**Parameters**:
- `upper` (Union[Array, np.ndarray]): Upper surface coordinates (2, n_upper) where first row is x, second is y
- `lower` (Union[Array, np.ndarray]): Lower surface coordinates (2, n_lower) where first row is x, second is y
- `name` (str): Name of the airfoil. Default: "JaxAirfoil"
- `buffer_size` (Optional[int]): Optional buffer size (auto-determined if None)
- `metadata` (Optional[Dict[str, Any]]): Optional additional metadata

**Returns**: JaxAirfoil instance

**Example**:
```python
# Define upper and lower surfaces separately
upper = jnp.array([[1.0, 0.5, 0.0], [0.0, 0.05, 0.08]])
lower = jnp.array([[0.0, 0.5, 1.0], [0.08, -0.05, 0.0]])

airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="CustomAirfoil")
```

##### `from_file(filename)`

```python
@classmethod
def from_file(cls, filename: str) -> "JaxAirfoil"
```

**Description**: Initialize the JaxAirfoil class from a file.

**Parameters**:
- `filename` (str): Name of the file to load the airfoil from

**Returns**: JaxAirfoil instance loaded from file

**Raises**:
- `ValueError`: If file cannot be parsed or contains invalid data
- `FileNotFoundError`: If file does not exist

**Example**:
```python
# Load airfoil from file
airfoil = JaxAirfoil.from_file("path/to/airfoil.dat")
```

#### Morphing Methods

##### `morph_new_from_two_foils(airfoil1, airfoil2, eta, n_points)`

```python
@classmethod
def morph_new_from_two_foils(
    cls,
    airfoil1: "JaxAirfoil",
    airfoil2: "JaxAirfoil",
    eta: float,
    n_points: int,
) -> "JaxAirfoil"
```

**Description**: Returns a new airfoil morphed between two airfoils. This method is fully differentiable with respect to the morphing parameter eta.

**Parameters**:
- `airfoil1` (JaxAirfoil): First airfoil (eta=0)
- `airfoil2` (JaxAirfoil): Second airfoil (eta=1)
- `eta` (float): Morphing parameter (0.0 = airfoil1, 1.0 = airfoil2)
- `n_points` (int): Number of points to generate

**Returns**: New JaxAirfoil morphed between the two airfoils

**Raises**:
- `ValueError`: If eta is not in range [0,1]

**Example**:
```python
# Create base airfoils
symmetric = JaxAirfoil.naca("0012", n_points=100)
cambered = JaxAirfoil.naca("4412", n_points=100)

# Morph between them
morphed = JaxAirfoil.morph_new_from_two_foils(
    symmetric, cambered, eta=0.5, n_points=150
)

# Use in gradient computation
def morphing_objective(eta):
    morphed = JaxAirfoil.morph_new_from_two_foils(
        symmetric, cambered, eta, n_points=100
    )
    return morphed.max_thickness

grad_fn = jax.grad(morphing_objective)
gradient = grad_fn(0.5)
```

### Instance Methods

#### Geometric Query Methods

##### `thickness(query_x)`

```python
def thickness(self, query_x: Union[Array, np.ndarray, float]) -> Array
```

**Description**: Compute airfoil thickness distribution at query points. Fully differentiable with respect to airfoil coordinates and query points.

**Parameters**:
- `query_x` (Union[Array, np.ndarray, float]): X coordinates to query thickness at

**Returns**: Thickness values at query points (Array)

**Example**:
```python
# Single point query
thickness_at_midchord = airfoil.thickness(0.5)

# Multiple points
x_query = jnp.linspace(0, 1, 20)
thickness_distribution = airfoil.thickness(x_query)

# Use in gradient computation
def thickness_objective(coords):
    airfoil = JaxAirfoil(coords)
    return jnp.mean(airfoil.thickness(jnp.array([0.25, 0.5, 0.75])))

grad_fn = jax.grad(thickness_objective)
```

##### `camber_line(query_x)`

```python
def camber_line(self, query_x: Union[Array, np.ndarray, float]) -> Array
```

**Description**: Compute airfoil camber line at query points. The camber line is the mean line between upper and lower surfaces.

**Parameters**:
- `query_x` (Union[Array, np.ndarray, float]): X coordinates to query camber line at

**Returns**: Camber line y-coordinates at query points (Array)

**Example**:
```python
# Camber at quarter chord
camber_25 = airfoil.camber_line(0.25)

# Camber distribution
x_query = jnp.linspace(0, 1, 50)
camber_dist = airfoil.camber_line(x_query)
```

##### `y_upper(query_x)`

```python
def y_upper(self, query_x: Union[Array, np.ndarray, float]) -> Array
```

**Description**: Query upper surface y-coordinates at given x-coordinates using interpolation.

**Parameters**:
- `query_x` (Union[Array, np.ndarray, float]): X coordinates to query

**Returns**: Upper surface y-coordinates at query points (Array)

**Example**:
```python
# Upper surface at specific points
x_points = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
y_upper = airfoil.y_upper(x_points)
```

##### `y_lower(query_x)`

```python
def y_lower(self, query_x: Union[Array, np.ndarray, float]) -> Array
```

**Description**: Query lower surface y-coordinates at given x-coordinates using interpolation.

**Parameters**:
- `query_x` (Union[Array, np.ndarray, float]): X coordinates to query

**Returns**: Lower surface y-coordinates at query points (Array)

**Example**:
```python
# Lower surface at specific points
x_points = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
y_lower = airfoil.y_lower(x_points)
```

#### Transformation Methods

##### `flap(flap_hinge_chord_percentage, flap_angle, flap_hinge_thickness_percentage=0.5, chord_extension=1.0)`

```python
def flap(
    self,
    flap_hinge_chord_percentage: float,
    flap_angle: float,
    flap_hinge_thickness_percentage: float = 0.5,
    chord_extension: float = 1.0,
) -> "JaxAirfoil"
```

**Description**: Apply flap transformation to the airfoil. Creates a new JaxAirfoil with a flap applied at the specified hinge location and angle. Fully differentiable with respect to all parameters.

**Parameters**:
- `flap_hinge_chord_percentage` (float): Chordwise location of the flap hinge (0.0 to 1.0)
- `flap_angle` (float): Flap deflection angle in degrees (positive = downward)
- `flap_hinge_thickness_percentage` (float): Position of hinge through thickness (0.0 = lower surface, 1.0 = upper surface). Default: 0.5
- `chord_extension` (float): Scaling factor for flap chord length (1.0 = no extension). Default: 1.0

**Returns**: New JaxAirfoil instance with flap applied

**Raises**:
- `ValueError`: If parameters are out of valid ranges

**Example**:
```python
# Basic flap application
flapped = airfoil.flap(
    flap_hinge_chord_percentage=0.75,
    flap_angle=15.0
)

# Advanced flap with custom parameters
advanced_flap = airfoil.flap(
    flap_hinge_chord_percentage=0.8,
    flap_angle=20.0,
    flap_hinge_thickness_percentage=0.3,  # Closer to lower surface
    chord_extension=1.1  # 10% chord extension
)

# Use in optimization
def flap_objective(flap_angle):
    flapped = airfoil.flap(0.75, flap_angle)
    return flapped.max_camber

grad_fn = jax.grad(flap_objective)
gradient = grad_fn(15.0)
```

##### `repanel(n_points=200, distribution="cosine")`

```python
def repanel(
    self,
    n_points: int = 200,
    distribution: str = "cosine",
) -> "JaxAirfoil"
```

**Description**: Repanel the airfoil with a new point distribution. Redistributes points while preserving the airfoil shape and maintaining differentiability.

**Parameters**:
- `n_points` (int): Number of points for the repaneled airfoil. Default: 200
- `distribution` (str): Point distribution type - "cosine" or "uniform". Default: "cosine"

**Returns**: New JaxAirfoil instance with repaneled coordinates

**Raises**:
- `ValueError`: If distribution type is not supported

**Example**:
```python
# Repanel with cosine distribution
repaneled_cosine = airfoil.repanel(n_points=300, distribution="cosine")

# Repanel with uniform distribution
repaneled_uniform = airfoil.repanel(n_points=150, distribution="uniform")
```

#### File I/O Methods

##### `save_selig(directory=None, header=False, inverse=False)`

```python
def save_selig(
    self,
    directory: Optional[str] = None,
    header: bool = False,
    inverse: bool = False,
) -> None
```

**Description**: Saves the airfoil in the selig format.

**Parameters**:
- `directory` (Optional[str]): Directory to save the airfoil. If None, saves in current directory
- `header` (bool): Whether to include the header. Default: False
- `inverse` (bool): Whether to save the airfoil in the reverse selig format. Default: False

**Example**:
```python
# Basic save
airfoil.save_selig()

# Save with header in specific directory
airfoil.save_selig(directory="output/airfoils/", header=True)
```

##### `save_le(directory=None, header=False)`

```python
def save_le(
    self,
    directory: Optional[str] = None,
    header: bool = False,
) -> None
```

**Description**: Saves the airfoil in the reverse selig format (leading edge format).

**Parameters**:
- `directory` (Optional[str]): Directory to save the airfoil. If None, saves in current directory
- `header` (bool): Whether to include the header. Default: False

**Example**:
```python
# Save in leading edge format
airfoil.save_le(directory="output/", header=True)
```

#### Plotting Methods

##### `plot(camber=False, scatter=False, max_thickness=False, ax=None, **kwargs)`

```python
def plot(
    self,
    camber: bool = False,
    scatter: bool = False,
    max_thickness: bool = False,
    ax: Optional[Axes] = None,
    overide_color: Optional[str] = None,
    linewidth: float = 1.5,
    markersize: float = 2.0,
    alpha: float = 1.0,
    show_legend: bool = False,
) -> Optional[Axes]
```

**Description**: Plots the airfoil with various visualization options.

**Parameters**:
- `camber` (bool): Whether to plot the camber line. Default: False
- `scatter` (bool): Whether to plot the airfoil as a scatter plot. Default: False
- `max_thickness` (bool): Whether to plot the max thickness indicator. Default: False
- `ax` (Optional[Axes]): Matplotlib axes object. If None, creates new figure
- `overide_color` (Optional[str]): Override color for the plot
- `linewidth` (float): Line width for the plot. Default: 1.5
- `markersize` (float): Marker size for scatter plots. Default: 2.0
- `alpha` (float): Transparency level. Default: 1.0
- `show_legend` (bool): Whether to show legend. Default: False

**Returns**: Matplotlib axes object if ax was None, otherwise None

**Raises**:
- `ImportError`: If matplotlib is not available
- `ValueError`: If airfoil has no valid points

**Example**:
```python
import matplotlib.pyplot as plt

# Basic plot
airfoil.plot()

# Advanced plot with all features
fig, ax = plt.subplots()
airfoil.plot(
    ax=ax,
    camber=True,
    max_thickness=True,
    overide_color="purple",
    linewidth=2.0,
    show_legend=True
)

# Scatter plot
airfoil.plot(scatter=True, markersize=3.0, alpha=0.7)
```

### Properties

#### Basic Properties

##### `name`

```python
@property
def name(self) -> str
```

**Description**: Get the airfoil name.

**Returns**: Airfoil name (str)

**Example**:
```python
print(f"Airfoil name: {airfoil.name}")
```

##### `n_points`

```python
@property
def n_points(self) -> int
```

**Description**: Get the number of valid points in the airfoil.

**Returns**: Number of valid points (int)

**Example**:
```python
print(f"Number of points: {airfoil.n_points}")
```

##### `buffer_size`

```python
@property
def buffer_size(self) -> int
```

**Description**: Get the current buffer size used for static allocation.

**Returns**: Current buffer size (int)

**Example**:
```python
print(f"Buffer size: {airfoil.buffer_size}")
print(f"Memory efficiency: {airfoil.n_points/airfoil.buffer_size:.2%}")
```

#### Geometric Properties

##### `max_thickness`

```python
@property
def max_thickness(self) -> float
```

**Description**: Get the maximum thickness of the airfoil. Fully differentiable.

**Returns**: Maximum thickness value (float)

**Example**:
```python
print(f"Maximum thickness: {airfoil.max_thickness:.4f}")

# Use in gradient computation
def thickness_objective(coords):
    airfoil = JaxAirfoil(coords)
    return airfoil.max_thickness

grad_fn = jax.grad(thickness_objective)
```

##### `max_thickness_location`

```python
@property
def max_thickness_location(self) -> float
```

**Description**: Get the x-coordinate location of maximum thickness.

**Returns**: X-coordinate of maximum thickness (float)

**Example**:
```python
print(f"Max thickness at x/c = {airfoil.max_thickness_location:.3f}")
```

##### `max_camber`

```python
@property
def max_camber(self) -> float
```

**Description**: Get the maximum camber of the airfoil. Fully differentiable.

**Returns**: Maximum camber value (float)

**Example**:
```python
print(f"Maximum camber: {airfoil.max_camber:.4f}")
```

##### `max_camber_location`

```python
@property
def max_camber_location(self) -> float
```

**Description**: Get the x-coordinate location of maximum camber.

**Returns**: X-coordinate of maximum camber (float)

**Example**:
```python
print(f"Max camber at x/c = {airfoil.max_camber_location:.3f}")
```

##### `chord_length`

```python
@property
def chord_length(self) -> float
```

**Description**: Get the chord length of the airfoil (distance from leading edge to trailing edge).

**Returns**: Chord length (float)

**Example**:
```python
print(f"Chord length: {airfoil.chord_length:.4f}")
```

#### Coordinate Properties

##### `upper_surface_points`

```python
@property
def upper_surface_points(self) -> Tuple[Array, Array]
```

**Description**: Get the upper surface points (x, y) of the airfoil.

**Returns**: Tuple of (x_upper, y_upper) arrays

**Example**:
```python
x_upper, y_upper = airfoil.upper_surface_points
print(f"Upper surface has {len(x_upper)} points")
```

##### `lower_surface_points`

```python
@property
def lower_surface_points(self) -> Tuple[Array, Array]
```

**Description**: Get the lower surface points (x, y) of the airfoil.

**Returns**: Tuple of (x_lower, y_lower) arrays

**Example**:
```python
x_lower, y_lower = airfoil.lower_surface_points
print(f"Lower surface has {len(x_lower)} points")
```

### Batch Processing Methods

#### `create_batch_from_list(airfoils, target_buffer_size=None)`

```python
@classmethod
def create_batch_from_list(
    cls,
    airfoils: List["JaxAirfoil"],
    target_buffer_size: Optional[int] = None,
) -> Tuple[
    Float[Array, "batch_size 2 buffer_size"],
    Bool[Array, "batch_size buffer_size"],
    List[int],
    List[int],
]
```

**Description**: Create batch arrays from a list of JaxAirfoil instances for efficient vectorized operations.

**Parameters**:
- `airfoils` (List[JaxAirfoil]): List of JaxAirfoil instances to batch
- `target_buffer_size` (Optional[int]): Optional target buffer size (auto-determined if None)

**Returns**: Tuple of (batch_coords, batch_validity_masks, upper_split_indices, n_valid_points)

**Example**:
```python
# Create multiple airfoils
airfoils = [
    JaxAirfoil.naca("2412", n_points=100),
    JaxAirfoil.naca("0012", n_points=100),
    JaxAirfoil.naca("4412", n_points=100)
]

# Create batch
batch_coords, batch_masks, upper_splits, n_valid = JaxAirfoil.create_batch_from_list(airfoils)
print(f"Batch shape: {batch_coords.shape}")
```

#### `batch_naca4(digits_list, n_points=200, target_buffer_size=None)`

```python
@classmethod
def batch_naca4(
    cls,
    digits_list: List[str],
    n_points: int = 200,
    target_buffer_size: Optional[int] = None,
) -> Tuple[
    Float[Array, "batch_size 2 buffer_size"],
    Bool[Array, "batch_size buffer_size"],
    List[int],
    List[int],
]
```

**Description**: Create a batch of NACA 4-digit airfoils efficiently.

**Parameters**:
- `digits_list` (List[str]): List of NACA 4-digit designations (e.g., ["2412", "0012"])
- `n_points` (int): Number of points for each surface. Default: 200
- `target_buffer_size` (Optional[int]): Optional target buffer size (auto-determined if None)

**Returns**: Tuple of (batch_coords, batch_validity_masks, upper_split_indices, n_valid_points)

**Example**:
```python
# Create batch of NACA airfoils
naca_codes = ["2412", "0012", "4412", "6412"]
batch_coords, batch_masks, upper_splits, n_valid = JaxAirfoil.batch_naca4(
    naca_codes, n_points=150
)
```

#### `batch_thickness(batch_coords, upper_split_indices, n_valid_points, query_x, target_buffer_size)`

```python
@staticmethod
def batch_thickness(
    batch_coords: Float[Array, "batch_size 2 buffer_size"],
    upper_split_indices: List[int],
    n_valid_points: List[int],
    query_x: Union[Array, np.ndarray, float],
    target_buffer_size: int,
) -> Float[Array, "batch_size n_query"]
```

**Description**: Compute thickness distribution for a batch of airfoils efficiently.

**Parameters**:
- `batch_coords` (Array): Batch coordinate arrays (batch_size, 2, buffer_size)
- `upper_split_indices` (List[int]): List of upper surface split indices for each airfoil
- `n_valid_points` (List[int]): List of valid point counts for each airfoil
- `query_x` (Union[Array, np.ndarray, float]): X coordinates to query thickness at
- `target_buffer_size` (int): Buffer size used for the batch

**Returns**: Thickness values for each airfoil at query points (batch_size, n_query)

**Example**:
```python
# Compute thickness for batch
query_x = jnp.linspace(0, 1, 20)
batch_thickness = JaxAirfoil.batch_thickness(
    batch_coords, upper_splits, n_valid, query_x, batch_coords.shape[2]
)
print(f"Batch thickness shape: {batch_thickness.shape}")
```

### JAX Integration

#### Pytree Registration

The JaxAirfoil class is registered as a JAX pytree, enabling it to work seamlessly with JAX transformations:

```python
# Works with jax.jit
@jax.jit
def compute_properties(airfoil):
    return airfoil.max_thickness, airfoil.max_camber

# Works with jax.grad
def objective(airfoil_coords):
    airfoil = JaxAirfoil(airfoil_coords)
    return airfoil.max_thickness

grad_fn = jax.grad(objective)

# Works with jax.vmap
@jax.vmap
def batch_properties(airfoil_coords_batch):
    airfoil = JaxAirfoil(airfoil_coords_batch)
    return airfoil.max_thickness
```

#### Tree Flatten/Unflatten Methods

##### `tree_flatten()`

```python
def tree_flatten(self) -> Tuple[Tuple[Array, ...], Dict[str, Any]]
```

**Description**: Flatten the JaxAirfoil instance for JAX pytree registration. Required for JAX transformations.

**Returns**: Tuple of (children, aux_data) where children are differentiable arrays and aux_data is non-differentiable metadata

##### `tree_unflatten(aux_data, children)`

```python
@classmethod
def tree_unflatten(
    cls, aux_data: Dict[str, Any], children: Tuple[Array, ...]
) -> "JaxAirfoil"
```

**Description**: Unflatten a JaxAirfoil instance from flattened data. Required for JAX pytree registration.

**Parameters**:
- `aux_data` (Dict[str, Any]): Dictionary of auxiliary data (not differentiable)
- `children` (Tuple[Array, ...]): Tuple of arrays that are differentiable

**Returns**: Reconstructed JaxAirfoil instance

### Error Handling

The JAX Airfoil implementation includes comprehensive error handling with specific exception types:

#### Exception Types

- `AirfoilValidationError`: Raised for invalid airfoil parameters or geometry
- `GeometryError`: Raised for geometric computation errors
- `BufferOverflowError`: Raised when buffer capacity is exceeded

#### Validation Methods

All public methods include input validation with helpful error messages and suggested fixes:

```python
try:
    airfoil = JaxAirfoil.naca4("241")  # Invalid: too short
except AirfoilValidationError as e:
    print(f"Error: {e}")
    # Error includes suggested fixes
```

### Performance Considerations

#### JIT Compilation

- First calls to methods may be slower due to compilation
- Subsequent calls with same shapes are fast
- Use static arguments where possible for better compilation

#### Memory Management

- Uses static buffer allocation for efficiency
- Buffer sizes are powers of 2 for optimal memory usage
- Automatic recompilation when buffer overflow occurs

#### Gradient Computation

- All geometric operations support automatic differentiation
- Use `jax.grad` for scalar objectives
- Use `jax.jacrev` or `jax.jacfwd` for vector objectives

This API reference provides comprehensive documentation for all public methods and properties of the JaxAirfoil class, including usage examples and integration with JAX transformations.
