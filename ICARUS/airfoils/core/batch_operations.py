"""
Batch processing operations for JAX airfoil implementation.

This module provides vectorized operations for processing multiple airfoils simultaneously,
enabling efficient batch processing with JAX's vmap transformation. It includes batch
morphing, transformation functions, and efficient padding strategies for variable batch sizes.
"""

from functools import partial
from typing import List
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int

from .buffer_management import AirfoilBufferManager
from .operations import JaxAirfoilOps


class BatchAirfoilOps:
    """
    Batch processing operations for multiple airfoils.

    This class provides vectorized operations that can process multiple airfoils
    simultaneously using JAX's vmap transformation. All operations are designed
    to work with padded arrays and masking to handle variable-sized airfoil data
    efficiently in batch mode.

    The batch operations support:
    - Batch morphing between pairs of airfoils
    - Batch geometric transformations (flaps, scaling, rotation)
    - Batch interpolation and surface queries
    - Efficient padding strategies for variable batch sizes
    """

    @staticmethod
    def determine_batch_buffer_size(airfoil_sizes: List[int]) -> int:
        """
        Determine the optimal buffer size for a batch of airfoils.

        Args:
            airfoil_sizes: List of point counts for each airfoil in the batch

        Returns:
            Buffer size that can accommodate all airfoils in the batch
        """
        if not airfoil_sizes:
            return AirfoilBufferManager.MIN_BUFFER_SIZE

        max_size = max(airfoil_sizes)
        return AirfoilBufferManager.determine_buffer_size(max_size)

    @staticmethod
    def pad_batch_coordinates(
        batch_coords: List[Float[Array, "2 n_points"]],
        target_size: int,
    ) -> Tuple[
        Float[Array, "batch_size 2 target_size"],
        Bool[Array, "batch_size target_size"],
    ]:
        """
        Pad a batch of coordinate arrays to uniform size.

        Args:
            batch_coords: List of coordinate arrays, each of shape (2, n_points_i)
            target_size: Target buffer size for all arrays

        Returns:
            Tuple of (padded_batch_coords, batch_validity_masks) where:
                - padded_batch_coords: Shape (batch_size, 2, target_size)
                - batch_validity_masks: Shape (batch_size, target_size)
        """
        batch_size = len(batch_coords)

        # Initialize batch arrays
        padded_batch = jnp.full((batch_size, 2, target_size), jnp.nan)
        validity_masks = jnp.zeros((batch_size, target_size), dtype=bool)

        # Process each airfoil in the batch
        for i, coords in enumerate(batch_coords):
            n_points = coords.shape[1]

            # Pad coordinates
            padded_coords = AirfoilBufferManager.pad_coordinates(coords, target_size)

            # Create validity mask
            validity_mask = AirfoilBufferManager.create_validity_mask(
                n_points,
                target_size,
            )

            # Store in batch arrays
            padded_batch = padded_batch.at[i].set(padded_coords)
            validity_masks = validity_masks.at[i].set(validity_mask)

        return padded_batch, validity_masks

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_compute_thickness(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, " n_query"],
    ) -> Float[Array, "batch_size n_query"]:
        """
        Compute thickness distribution for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)
            query_x: X coordinates to query thickness at

        Returns:
            Thickness values for each airfoil at query points (batch_size, n_query)
        """
        # Use vmap to vectorize the single-airfoil thickness computation
        vectorized_thickness = jax.vmap(
            JaxAirfoilOps.compute_thickness,
            in_axes=(
                0,
                0,
                None,
                None,
                None,
            ),  # Batch over first two args, broadcast others
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_thickness(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
            query_x,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_compute_camber_line(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, " n_query"],
    ) -> Float[Array, "batch_size n_query"]:
        """
        Compute camber line for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)
            query_x: X coordinates to query camber line at

        Returns:
            Camber line values for each airfoil at query points (batch_size, n_query)
        """
        # Use vmap to vectorize the single-airfoil camber line computation
        vectorized_camber = jax.vmap(
            JaxAirfoilOps.compute_camber_line,
            in_axes=(
                0,
                0,
                None,
                None,
                None,
            ),  # Batch over first two args, broadcast others
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_camber(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
            query_x,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def batch_y_upper(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        n_valid: int,
        query_x: Float[Array, " n_query"],
    ) -> Float[Array, "batch_size n_query"]:
        """
        Query upper surface y-coordinates for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            n_valid: Number of valid points (same for all airfoils)
            query_x: X coordinates to query

        Returns:
            Upper surface y-coordinates for each airfoil at query points (batch_size, n_query)
        """
        # Use vmap to vectorize the single-airfoil y_upper computation
        vectorized_y_upper = jax.vmap(
            JaxAirfoilOps.y_upper,
            in_axes=(0, None, None),  # Batch over first arg, broadcast others
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_y_upper(batch_upper_coords, n_valid, query_x)

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def batch_y_lower(
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_valid: int,
        query_x: Float[Array, " n_query"],
    ) -> Float[Array, "batch_size n_query"]:
        """
        Query lower surface y-coordinates for a batch of airfoils.

        Args:
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_valid: Number of valid points (same for all airfoils)
            query_x: X coordinates to query

        Returns:
            Lower surface y-coordinates for each airfoil at query points (batch_size, n_query)
        """
        # Use vmap to vectorize the single-airfoil y_lower computation
        vectorized_y_lower = jax.vmap(
            JaxAirfoilOps.y_lower,
            in_axes=(0, None, None),  # Batch over first arg, broadcast others
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_y_lower(batch_lower_coords, n_valid, query_x)

    @staticmethod
    @jax.jit
    def batch_morph_airfoils(
        batch_coords1: Float[Array, "batch_size 2 n_buffer"],
        batch_coords2: Float[Array, "batch_size 2 n_buffer"],
        batch_validity_masks1: Bool[Array, "batch_size n_buffer"],
        batch_validity_masks2: Bool[Array, "batch_size n_buffer"],
        eta: Float[Array, " batch_size"],
    ) -> Float[Array, "batch_size 2 n_buffer"]:
        """
        Morph between pairs of airfoils in batch mode.

        Args:
            batch_coords1: First batch of airfoil coordinates (batch_size, 2, n_buffer)
            batch_coords2: Second batch of airfoil coordinates (batch_size, 2, n_buffer)
            batch_validity_masks1: Validity masks for first batch (batch_size, n_buffer)
            batch_validity_masks2: Validity masks for second batch (batch_size, n_buffer)
            eta: Morphing parameters for each pair (batch_size,) where 0=first, 1=second

        Returns:
            Morphed airfoil coordinates (batch_size, 2, n_buffer)
        """
        # Expand eta to match coordinate dimensions
        eta_expanded = eta[:, None, None]  # Shape: (batch_size, 1, 1)

        # Perform linear interpolation between the two coordinate sets
        # Only interpolate where both airfoils have valid points
        combined_mask = batch_validity_masks1 & batch_validity_masks2

        # Linear interpolation: result = (1 - eta) * coords1 + eta * coords2
        morphed_coords = (
            1.0 - eta_expanded
        ) * batch_coords1 + eta_expanded * batch_coords2

        # Apply combined validity mask to ensure we only have valid results
        morphed_coords = jnp.where(
            combined_mask[:, None, :],  # Expand mask to match coordinate dimensions
            morphed_coords,
            jnp.nan,
        )

        return morphed_coords

    @staticmethod
    @jax.jit
    def batch_rotate_coordinates(
        batch_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_angles: Float[Array, " batch_size"],
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> Float[Array, "batch_size 2 n_buffer"]:
        """
        Rotate coordinates for a batch of airfoils.

        Args:
            batch_coords: Batch of coordinates to rotate (batch_size, 2, n_buffer)
            batch_angles: Rotation angles in radians for each airfoil (batch_size,)
            center_x: X coordinate of rotation center
            center_y: Y coordinate of rotation center

        Returns:
            Rotated coordinates with same shape as input
        """
        # Use vmap to vectorize the single-airfoil rotation
        vectorized_rotate = jax.vmap(
            JaxAirfoilOps.rotate_coordinates,
            in_axes=(
                0,
                0,
                None,
                None,
            ),  # Batch over coords and angles, broadcast centers
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_rotate(batch_coords, batch_angles, center_x, center_y)

    @staticmethod
    @jax.jit
    def batch_scale_coordinates(
        batch_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_scale_x: Float[Array, " batch_size"],
        batch_scale_y: Float[Array, " batch_size"],
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> Float[Array, "batch_size 2 n_buffer"]:
        """
        Scale coordinates for a batch of airfoils.

        Args:
            batch_coords: Batch of coordinates to scale (batch_size, 2, n_buffer)
            batch_scale_x: Scaling factors in x direction for each airfoil (batch_size,)
            batch_scale_y: Scaling factors in y direction for each airfoil (batch_size,)
            center_x: X coordinate of scaling center
            center_y: Y coordinate of scaling center

        Returns:
            Scaled coordinates with same shape as input
        """
        # Use vmap to vectorize the single-airfoil scaling
        vectorized_scale = jax.vmap(
            JaxAirfoilOps.scale_coordinates,
            in_axes=(
                0,
                0,
                0,
                None,
                None,
            ),  # Batch over coords and scales, broadcast centers
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_scale(
            batch_coords,
            batch_scale_x,
            batch_scale_y,
            center_x,
            center_y,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_apply_flap_transformation(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        batch_flap_hinge_x: Float[Array, " batch_size"],
        batch_flap_angles: Float[Array, " batch_size"],
        flap_hinge_thickness_percentage: float = 0.5,
        chord_extension: float = 1.0,
    ) -> Tuple[
        Float[Array, "batch_size 2 n_buffer"],
        Float[Array, "batch_size 2 n_buffer"],
        Int[Array, " batch_size"],
        Int[Array, " batch_size"],
    ]:
        """
        Apply flap transformation to a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)
            batch_flap_hinge_x: X-coordinates of flap hinges for each airfoil (batch_size,)
            batch_flap_angles: Flap deflection angles in radians for each airfoil (batch_size,)
            flap_hinge_thickness_percentage: Position of hinge through thickness (0=lower, 1=upper)
            chord_extension: Scaling factor for flap chord length

        Returns:
            Tuple of (new_upper_coords, new_lower_coords, new_n_upper_valid, new_n_lower_valid)
            where each array has batch dimension
        """
        # Create should_skip flags for each airfoil
        batch_should_skip = (batch_flap_angles == 0.0) | (batch_flap_hinge_x >= 1.0)

        # Use vmap to vectorize the single-airfoil flap transformation
        vectorized_flap = jax.vmap(
            JaxAirfoilOps.apply_flap_transformation,
            in_axes=(0, 0, None, None, 0, 0, None, None, 0),  # Batch over varying args
            out_axes=(0, 0, 0, 0),  # All outputs have batch dimension
        )

        return vectorized_flap(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
            batch_flap_hinge_x,
            batch_flap_angles,
            flap_hinge_thickness_percentage,
            chord_extension,
            batch_should_skip,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def batch_generate_naca4_coordinates(
        batch_max_camber: Float[Array, " batch_size"],
        batch_camber_position: Float[Array, " batch_size"],
        batch_thickness: Float[Array, " batch_size"],
        n_points: int,
    ) -> Tuple[
        Float[Array, "batch_size 2 n_points"],
        Float[Array, "batch_size 2 n_points"],
    ]:
        """
        Generate NACA 4-digit airfoil coordinates for a batch of parameter sets.

        Args:
            batch_max_camber: Maximum camber values for each airfoil (batch_size,)
            batch_camber_position: Camber position values for each airfoil (batch_size,)
            batch_thickness: Thickness values for each airfoil (batch_size,)
            n_points: Number of points for each surface (same for all airfoils)

        Returns:
            Tuple of (batch_upper_coords, batch_lower_coords) where each has shape
            (batch_size, 2, n_points)
        """
        # Use vmap to vectorize the single-airfoil NACA generation
        vectorized_naca4 = jax.vmap(
            JaxAirfoilOps.generate_naca4_coordinates,
            in_axes=(0, 0, 0, None),  # Batch over parameters, broadcast n_points
            out_axes=(0, 0),  # Both outputs have batch dimension
        )

        return vectorized_naca4(
            batch_max_camber,
            batch_camber_position,
            batch_thickness,
            n_points,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def batch_generate_naca5_coordinates(
        batch_design_cl: Float[Array, " batch_size"],
        batch_max_camber_pos: Float[Array, " batch_size"],
        batch_reflex: Bool[Array, " batch_size"],
        batch_thickness: Float[Array, " batch_size"],
        n_points: int,
    ) -> Tuple[
        Float[Array, "batch_size 2 n_points"],
        Float[Array, "batch_size 2 n_points"],
    ]:
        """
        Generate NACA 5-digit airfoil coordinates for a batch of parameter sets.

        Args:
            batch_design_cl: Design coefficient of lift values for each airfoil (batch_size,)
            batch_max_camber_pos: Maximum camber position values for each airfoil (batch_size,)
            batch_reflex: Reflex flags for each airfoil (batch_size,)
            batch_thickness: Thickness values for each airfoil (batch_size,)
            n_points: Number of points for each surface (same for all airfoils)

        Returns:
            Tuple of (batch_upper_coords, batch_lower_coords) where each has shape
            (batch_size, 2, n_points)
        """
        # Use vmap to vectorize the single-airfoil NACA generation
        vectorized_naca5 = jax.vmap(
            JaxAirfoilOps.generate_naca5_coordinates,
            in_axes=(0, 0, 0, 0, None),  # Batch over parameters, broadcast n_points
            out_axes=(0, 0),  # Both outputs have batch dimension
        )

        return vectorized_naca5(
            batch_design_cl,
            batch_max_camber_pos,
            batch_reflex,
            batch_thickness,
            n_points,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_compute_max_thickness(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        n_sample_points: int = 100,
    ) -> Tuple[Float[Array, " batch_size"], Float[Array, " batch_size"]]:
        """
        Compute maximum thickness and location for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)
            n_sample_points: Number of points to sample for thickness computation

        Returns:
            Tuple of (max_thickness_values, max_thickness_locations) where each has shape (batch_size,)
        """
        # Use vmap to vectorize the single-airfoil max thickness computation
        vectorized_max_thickness = jax.vmap(
            JaxAirfoilOps.compute_max_thickness,
            in_axes=(0, 0, None, None, None),  # Batch over coords, broadcast others
            out_axes=(0, 0),  # Both outputs have batch dimension
        )

        return vectorized_max_thickness(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
            n_sample_points,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_compute_max_camber(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        n_sample_points: int = 100,
    ) -> Tuple[Float[Array, " batch_size"], Float[Array, " batch_size"]]:
        """
        Compute maximum camber and location for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)
            n_sample_points: Number of points to sample for camber computation

        Returns:
            Tuple of (max_camber_values, max_camber_locations) where each has shape (batch_size,)
        """
        # Use vmap to vectorize the single-airfoil max camber computation
        vectorized_max_camber = jax.vmap(
            JaxAirfoilOps.compute_max_camber,
            in_axes=(0, 0, None, None, None),  # Batch over coords, broadcast others
            out_axes=(0, 0),  # Both outputs have batch dimension
        )

        return vectorized_max_camber(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
            n_sample_points,
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3))
    def batch_compute_chord_length(
        batch_upper_coords: Float[Array, "batch_size 2 n_buffer"],
        batch_lower_coords: Float[Array, "batch_size 2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
    ) -> Float[Array, " batch_size"]:
        """
        Compute chord length for a batch of airfoils.

        Args:
            batch_upper_coords: Batch of upper surface coordinates (batch_size, 2, n_buffer)
            batch_lower_coords: Batch of lower surface coordinates (batch_size, 2, n_buffer)
            n_upper_valid: Number of valid upper surface points (same for all airfoils)
            n_lower_valid: Number of valid lower surface points (same for all airfoils)

        Returns:
            Chord lengths for each airfoil (batch_size,)
        """
        # Use vmap to vectorize the single-airfoil chord length computation
        vectorized_chord_length = jax.vmap(
            JaxAirfoilOps.compute_chord_length,
            in_axes=(0, 0, None, None),  # Batch over coords, broadcast others
            out_axes=0,  # Output has batch dimension
        )

        return vectorized_chord_length(
            batch_upper_coords,
            batch_lower_coords,
            n_upper_valid,
            n_lower_valid,
        )
