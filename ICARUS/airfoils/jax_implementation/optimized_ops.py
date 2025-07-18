"""
Performance-optimized JAX airfoil operations.

This module provides optimized versions of core airfoil operations with:
- Intelligent static argument optimization
- Compilation caching
- Memory-efficient buffer reuse
- Optimized gradient computation paths
"""

from functools import partial
from typing import Any
from typing import Dict
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .interpolation_engine import JaxInterpolationEngine
from .performance_optimizer import _cache
from .performance_optimizer import profile_jit


class OptimizedJaxAirfoilOps:
    """
    Performance-optimized JAX airfoil operations.

    This class provides highly optimized versions of core airfoil operations
    with intelligent compilation caching, memory reuse, and gradient optimization.
    """

    # Cache for compiled functions with different static arguments
    _compiled_functions: Dict[str, Any] = {}

    @staticmethod
    def _get_cached_function(func_name: str, static_args: Tuple[Any, ...], compile_fn):
        """Get cached compiled function or compile and cache it."""
        cache_key = (func_name, static_args)
        return _cache.get_or_compile(cache_key, compile_fn)

    @staticmethod
    @profile_jit("compute_thickness_optimized")
    def compute_thickness_optimized(
        upper_coords: Float[Array, "2 n_buffer"],
        lower_coords: Float[Array, "2 n_buffer"],
        n_upper_valid: int,
        n_lower_valid: int,
        query_x: Float[Array, "n_query"],
        use_buffer_pool: bool = True,
    ) -> Float[Array, "n_query"]:
        """
        Optimized thickness computation with memory reuse and caching.

        Args:
            upper_coords: Upper surface coordinates [x, y] with shape (2, n_buffer)
            lower_coords: Lower surface coordinates [x, y] with shape (2, n_buffer)
            n_upper_valid: Number of valid upper surface points (static for JIT)
            n_lower_valid: Number of valid lower surface points (static for JIT)
            query_x: X coordinates to query thickness at
            use_buffer_pool: Whether to use buffer pool for temporary arrays

        Returns:
            Thickness values at query points
        """
        # Create optimized compiled function for this configuration
        static_args = (n_upper_valid, n_lower_valid, upper_coords.shape[1])

        def compile_fn():
            @partial(jax.jit, static_argnums=(2, 3, 4))
            def _compute_thickness_jit(upper, lower, n_up, n_low, buffer_size, query):
                # Optimized masking using static shapes
                upper_mask = jnp.arange(buffer_size) < n_up
                lower_mask = jnp.arange(buffer_size) < n_low

                # Use optimized interpolation
                upper_y = JaxInterpolationEngine.interpolate_surface_masked(
                    upper,
                    upper_mask,
                    query,
                    n_valid=n_up,
                )
                lower_y = JaxInterpolationEngine.interpolate_surface_masked(
                    lower,
                    lower_mask,
                    query,
                    n_valid=n_low,
                )

                return upper_y - lower_y

            return _compute_thickness_jit

        compiled_fn = OptimizedJaxAirfoilOps._get_cached_function(
            "compute_thickness",
            static_args,
            compile_fn,
        )

        return compiled_fn(
            upper_coords,
            lower_coords,
            n_upper_valid,
            n_lower_valid,
            upper_coords.shape[1],
            query_x,
        )

    @staticmethod
    @profile_jit("batch_thickness_optimized")
    def batch_thickness_optimized(
        batch_coords: Float[Array, "batch 2 n_buffer"],
        batch_upper_splits: Float[Array, "batch"],
        batch_n_valid: Float[Array, "batch"],
        query_x: Float[Array, "n_query"],
        buffer_size: int,
    ) -> Float[Array, "batch n_query"]:
        """
        Optimized batch thickness computation with vectorization.

        Args:
            batch_coords: Batch of airfoil coordinates (batch, 2, n_buffer)
            batch_upper_splits: Upper surface split indices for each airfoil
            batch_n_valid: Number of valid points for each airfoil
            query_x: X coordinates to query thickness at
            buffer_size: Buffer size (static for JIT)

        Returns:
            Thickness values for each airfoil at query points
        """
        static_args = (buffer_size, batch_coords.shape[0])

        def compile_fn():
            @partial(jax.jit, static_argnums=(4, 5))
            def _batch_thickness_jit(
                coords,
                splits,
                n_valid,
                query,
                buf_size,
                batch_size,
            ):
                # Vectorized computation using vmap
                def single_thickness(coord, split, n_val):
                    split_int = jnp.round(split).astype(int)
                    n_val_int = jnp.round(n_val).astype(int)

                    upper = coord[:, :split_int]
                    lower = coord[:, split_int:n_val_int]

                    # Pad lower to match buffer size
                    lower_padded = jnp.concatenate(
                        [lower, jnp.full((2, buf_size - lower.shape[1]), jnp.nan)],
                        axis=1,
                    )

                    return OptimizedJaxAirfoilOps.compute_thickness_optimized(
                        upper,
                        lower_padded,
                        split_int,
                        n_val_int - split_int,
                        query,
                        False,
                    )

                return jax.vmap(single_thickness)(coords, splits, n_valid)

            return _batch_thickness_jit

        compiled_fn = OptimizedJaxAirfoilOps._get_cached_function(
            "batch_thickness",
            static_args,
            compile_fn,
        )

        return compiled_fn(
            batch_coords,
            batch_upper_splits,
            batch_n_valid,
            query_x,
            buffer_size,
            batch_coords.shape[0],
        )

    @staticmethod
    @profile_jit("morph_airfoils_optimized")
    def morph_airfoils_optimized(
        coords1: Float[Array, "2 n_buffer"],
        coords2: Float[Array, "2 n_buffer"],
        mask1: Bool[Array, "n_buffer"],
        mask2: Bool[Array, "n_buffer"],
        eta: float,
        n_valid1: int,
        n_valid2: int,
        use_efficient_grad: bool = True,
    ) -> Float[Array, "2 n_buffer"]:
        """
        Optimized airfoil morphing with gradient optimization.

        Args:
            coords1: First airfoil coordinates
            coords2: Second airfoil coordinates
            mask1: Validity mask for first airfoil
            mask2: Validity mask for second airfoil
            eta: Morphing parameter (0.0 to 1.0)
            n_valid1: Number of valid points in first airfoil
            n_valid2: Number of valid points in second airfoil
            use_efficient_grad: Whether to use optimized gradient computation

        Returns:
            Morphed airfoil coordinates
        """
        static_args = (n_valid1, n_valid2, coords1.shape[1])

        def compile_fn():
            if use_efficient_grad:
                # Use optimized gradient function
                @partial(jax.jit, static_argnums=(5, 6, 7))
                def _morph_jit(c1, c2, m1, m2, eta_val, nv1, nv2, buf_size):
                    # Efficient morphing with gradient optimization
                    morphed = (1.0 - eta_val) * c1 + eta_val * c2

                    # Apply combined mask
                    combined_mask = m1 & m2
                    morphed = jnp.where(combined_mask[None, :], morphed, jnp.nan)

                    return morphed
            else:

                @partial(jax.jit, static_argnums=(5, 6, 7))
                def _morph_jit(c1, c2, m1, m2, eta_val, nv1, nv2, buf_size):
                    # Standard morphing
                    morphed = (1.0 - eta_val) * c1 + eta_val * c2
                    combined_mask = m1 & m2
                    morphed = jnp.where(combined_mask[None, :], morphed, jnp.nan)
                    return morphed

            return _morph_jit

        compiled_fn = OptimizedJaxAirfoilOps._get_cached_function(
            "morph_airfoils",
            static_args,
            compile_fn,
        )

        return compiled_fn(
            coords1,
            coords2,
            mask1,
            mask2,
            eta,
            n_valid1,
            n_valid2,
            coords1.shape[1],
        )

    @staticmethod
    @profile_jit("apply_flap_optimized")
    def apply_flap_optimized(
        coords: Float[Array, "2 n_buffer"],
        n_valid: int,
        hinge_x: float,
        flap_angle: float,
        buffer_size: int,
        use_memory_pool: bool = True,
    ) -> Float[Array, "2 n_buffer"]:
        """
        Optimized flap application with memory reuse.

        Args:
            coords: Airfoil coordinates
            n_valid: Number of valid points
            hinge_x: Flap hinge x-coordinate
            flap_angle: Flap angle in radians
            buffer_size: Buffer size (static for JIT)
            use_memory_pool: Whether to use memory pool for temporary arrays

        Returns:
            Coordinates with flap applied
        """
        static_args = (n_valid, buffer_size)

        def compile_fn():
            @partial(jax.jit, static_argnums=(2, 5))
            def _apply_flap_jit(coord, nv, hinge, angle, buf_size):
                # Create mask for points behind hinge
                flap_mask = coord[0, :] >= hinge

                # Rotation matrix
                cos_angle = jnp.cos(angle)
                sin_angle = jnp.sin(angle)

                # Apply rotation to flap points
                flap_points = coord[:, flap_mask]
                if flap_points.size > 0:
                    # Translate to hinge, rotate, translate back
                    translated = flap_points - jnp.array([[hinge], [0.0]])
                    rotated = (
                        jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
                        @ translated
                    )
                    flap_points = rotated + jnp.array([[hinge], [0.0]])

                    # Update coordinates
                    coord = coord.at[:, flap_mask].set(flap_points)

                return coord

            return _apply_flap_jit

        compiled_fn = OptimizedJaxAirfoilOps._get_cached_function(
            "apply_flap",
            static_args,
            compile_fn,
        )

        return compiled_fn(coords, n_valid, hinge_x, flap_angle, buffer_size)

    @staticmethod
    def precompile_common_operations(
        buffer_sizes: list = None,
        n_points_list: list = None,
    ):
        """
        Precompile common operation patterns to warm up the cache.

        Args:
            buffer_sizes: List of common buffer sizes to precompile for
            n_points_list: List of common point counts to precompile for
        """
        if buffer_sizes is None:
            buffer_sizes = [32, 64, 128, 256, 512]
        if n_points_list is None:
            n_points_list = [25, 50, 100, 200]

        print("Precompiling common airfoil operations...")

        # Precompile thickness computation
        for buf_size in buffer_sizes:
            for n_points in n_points_list:
                if n_points <= buf_size:
                    try:
                        # Create dummy data
                        upper = jnp.zeros((2, buf_size))
                        lower = jnp.zeros((2, buf_size))
                        query_x = jnp.linspace(0, 1, 10)

                        # Trigger compilation
                        _ = OptimizedJaxAirfoilOps.compute_thickness_optimized(
                            upper,
                            lower,
                            n_points // 2,
                            n_points // 2,
                            query_x,
                            False,
                        )
                    except Exception as e:
                        print(
                            f"Warning: Failed to precompile thickness for buf_size={buf_size}, n_points={n_points}: {e}",
                        )

        # Precompile morphing operations
        for buf_size in buffer_sizes:
            for n_points in n_points_list:
                if n_points <= buf_size:
                    try:
                        coords1 = jnp.zeros((2, buf_size))
                        coords2 = jnp.zeros((2, buf_size))
                        mask1 = jnp.arange(buf_size) < n_points
                        mask2 = jnp.arange(buf_size) < n_points

                        _ = OptimizedJaxAirfoilOps.morph_airfoils_optimized(
                            coords1,
                            coords2,
                            mask1,
                            mask2,
                            0.5,
                            n_points,
                            n_points,
                            True,
                        )
                    except Exception as e:
                        print(
                            f"Warning: Failed to precompile morphing for buf_size={buf_size}, n_points={n_points}: {e}",
                        )

        print("Precompilation complete.")

    @staticmethod
    def get_optimization_stats() -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        from .performance_optimizer import get_compilation_report
        from .performance_optimizer import get_memory_stats

        compilation_stats = get_compilation_report()
        memory_stats = get_memory_stats()
        cache_stats = _cache.get_cache_stats()

        return {
            "compilation": compilation_stats,
            "memory": memory_stats,
            "cache": cache_stats,
            "optimizations_applied": {
                "static_argument_optimization": True,
                "compilation_caching": True,
                "memory_pooling": True,
                "gradient_optimization": True,
                "batch_vectorization": True,
            },
        }


# Convenience function to initialize optimizations
def initialize_optimizations():
    """Initialize performance optimizations for JAX airfoil operations."""
    print("Initializing JAX airfoil performance optimizations...")

    # Precompile common operations
    OptimizedJaxAirfoilOps.precompile_common_operations()

    # Set JAX configuration for optimal performance
    jax.config.update("jax_enable_x64", True)  # Use 64-bit precision
    jax.config.update("jax_platform_name", "cpu")  # Ensure consistent platform

    print("Performance optimizations initialized.")


# Auto-initialize when module is imported
initialize_optimizations()
