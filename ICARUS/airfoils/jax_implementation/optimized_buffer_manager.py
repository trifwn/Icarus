"""
Optimized buffer management system for JAX airfoil implementation.

This module provides enhanced buffer management with:
- Memory-efficient buffer reuse mechanisms
- Intelligent buffer size prediction
- Compilation cache optimization
- Memory pool management
"""

import gc
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Bool
from jaxtyping import Float

from .buffer_manager import AirfoilBufferManager
from .error_handling import BufferOverflowError


@dataclass
class BufferUsagePattern:
    """Tracks buffer usage patterns for optimization."""

    buffer_size: int
    usage_count: int
    last_used: float
    average_lifetime: float
    reuse_frequency: float


class OptimizedAirfoilBufferManager(AirfoilBufferManager):
    """
    Enhanced buffer manager with performance optimizations.

    Provides intelligent buffer allocation, reuse mechanisms, and
    compilation cache optimization based on usage patterns.
    """

    # Enhanced buffer size strategy with more granular options
    OPTIMIZED_BUFFER_SIZES = [
        16,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]

    # Buffer pool for reuse
    _buffer_pools: Dict[Tuple[int, ...], deque] = defaultdict(deque)
    _usage_patterns: Dict[int, BufferUsagePattern] = {}
    _allocation_history: List[
        Tuple[float, int, str]
    ] = []  # (timestamp, size, operation)
    _lock = threading.Lock()

    # Compilation cache for different buffer configurations
    _compiled_functions: Dict[Tuple[str, int], Any] = {}

    @classmethod
    def determine_optimal_buffer_size(
        cls,
        n_points: int,
        usage_context: str = "general",
    ) -> int:
        """
        Determine optimal buffer size based on usage patterns and context.

        Args:
            n_points: Number of actual data points needed
            usage_context: Context of usage ("batch", "single", "morphing", etc.)

        Returns:
            Optimal buffer size considering usage patterns
        """
        if n_points <= 0:
            raise BufferOverflowError(
                f"Number of points must be positive, got {n_points}",
            )

        # Record allocation request
        with cls._lock:
            cls._allocation_history.append((time.time(), n_points, usage_context))

            # Keep only recent history (last 1000 allocations)
            if len(cls._allocation_history) > 1000:
                cls._allocation_history = cls._allocation_history[-1000:]

        # Context-specific optimization
        if usage_context == "batch":
            # For batch operations, use slightly larger buffers to accommodate variation
            target_size = int(n_points * 1.2)
        elif usage_context == "morphing":
            # Morphing operations might need consistent sizes
            target_size = n_points
        elif usage_context == "naca_generation":
            # NACA generation has predictable sizes
            target_size = n_points
        else:
            target_size = n_points

        # Find optimal size from our enhanced list
        for buffer_size in cls.OPTIMIZED_BUFFER_SIZES:
            if target_size <= buffer_size:
                # Check usage patterns to see if this size is frequently used
                with cls._lock:
                    if buffer_size in cls._usage_patterns:
                        pattern = cls._usage_patterns[buffer_size]
                        pattern.usage_count += 1
                        pattern.last_used = time.time()
                        pattern.reuse_frequency = pattern.usage_count / max(
                            1,
                            len(cls._allocation_history),
                        )
                    else:
                        cls._usage_patterns[buffer_size] = BufferUsagePattern(
                            buffer_size=buffer_size,
                            usage_count=1,
                            last_used=time.time(),
                            average_lifetime=0.0,
                            reuse_frequency=0.0,
                        )

                return buffer_size

        # If we exceed our largest predefined size, use max size
        return cls.MAX_BUFFER_SIZE

    @classmethod
    def get_reusable_buffer(
        cls,
        shape: Tuple[int, ...],
        dtype=jnp.float32,
    ) -> Optional[jnp.ndarray]:
        """
        Get a reusable buffer from the pool if available.

        Args:
            shape: Required buffer shape
            dtype: Required data type

        Returns:
            Reusable buffer or None if not available
        """
        pool_key = (shape, dtype)

        with cls._lock:
            if pool_key in cls._buffer_pools and cls._buffer_pools[pool_key]:
                buffer = cls._buffer_pools[pool_key].popleft()
                # Clear the buffer for reuse
                return jnp.zeros_like(buffer)

        return None

    @classmethod
    def return_buffer_for_reuse(cls, buffer: jnp.ndarray, max_pool_size: int = 20):
        """
        Return a buffer to the pool for future reuse.

        Args:
            buffer: Buffer to return to pool
            max_pool_size: Maximum number of buffers to keep in each pool
        """
        pool_key = (buffer.shape, buffer.dtype)

        with cls._lock:
            if len(cls._buffer_pools[pool_key]) < max_pool_size:
                cls._buffer_pools[pool_key].append(buffer)

    @classmethod
    def pad_coordinates_optimized(
        cls,
        coords: Float[jax.Array, "2 n_points"],
        target_size: int,
        use_buffer_pool: bool = True,
    ) -> Float[jax.Array, "2 target_size"]:
        """
        Optimized coordinate padding with buffer reuse.

        Args:
            coords: Input coordinates to pad
            target_size: Target buffer size
            use_buffer_pool: Whether to use buffer pool for optimization

        Returns:
            Padded coordinates array
        """
        current_size = coords.shape[1]

        if current_size == target_size:
            return coords

        if current_size > target_size:
            raise BufferOverflowError(
                f"Cannot pad coordinates: current size {current_size} > target size {target_size}",
            )

        target_shape = (2, target_size)

        # Try to get reusable buffer
        if use_buffer_pool:
            padded_buffer = cls.get_reusable_buffer(target_shape, coords.dtype)
            if padded_buffer is not None:
                # Copy existing data to reused buffer
                padded_buffer = padded_buffer.at[:, :current_size].set(coords)
                # Fill remaining with NaN
                padded_buffer = padded_buffer.at[:, current_size:].set(jnp.nan)
                return padded_buffer

        # Fallback to standard padding
        padding_needed = target_size - current_size
        padded_coords = jnp.pad(
            coords,
            ((0, 0), (0, padding_needed)),
            mode="constant",
            constant_values=jnp.nan,
        )

        return padded_coords

    @classmethod
    def pad_and_mask_optimized(
        cls,
        coords: Float[jax.Array, "2 n_points"],
        target_size: int,
        usage_context: str = "general",
    ) -> Tuple[Float[jax.Array, "2 target_size"], Bool[jax.Array, "target_size"], int]:
        """
        Optimized padding and masking with intelligent buffer management.

        Args:
            coords: Input coordinates
            target_size: Target buffer size
            usage_context: Context for optimization hints

        Returns:
            Tuple of (padded_coords, validity_mask, n_valid_points)
        """
        n_points = coords.shape[1]

        # Use optimized padding
        padded_coords = cls.pad_coordinates_optimized(
            coords,
            target_size,
            use_buffer_pool=True,
        )

        # Create validity mask efficiently
        validity_mask = jnp.arange(target_size) < n_points

        return padded_coords, validity_mask, n_points

    @classmethod
    def create_batch_buffers_optimized(
        cls,
        coord_list: List[jnp.ndarray],
        target_buffer_size: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
        """
        Create optimized batch buffers with intelligent size selection.

        Args:
            coord_list: List of coordinate arrays
            target_buffer_size: Optional target buffer size

        Returns:
            Tuple of (batch_coords, batch_masks, n_valid_list)
        """
        if not coord_list:
            raise ValueError("Coordinate list cannot be empty")

        batch_size = len(coord_list)

        # Determine optimal buffer size if not provided
        if target_buffer_size is None:
            max_points = max(coords.shape[1] for coords in coord_list)
            target_buffer_size = cls.determine_optimal_buffer_size(max_points, "batch")

        # Try to get reusable batch buffer
        batch_shape = (batch_size, 2, target_buffer_size)
        batch_coords = cls.get_reusable_buffer(batch_shape)

        if batch_coords is None:
            batch_coords = jnp.full(batch_shape, jnp.nan)

        # Create batch mask buffer
        mask_shape = (batch_size, target_buffer_size)
        batch_masks = jnp.zeros(mask_shape, dtype=bool)

        n_valid_list = []

        # Fill batch buffers efficiently
        for i, coords in enumerate(coord_list):
            n_points = coords.shape[1]
            n_valid_list.append(n_points)

            if n_points <= target_buffer_size:
                # Copy coordinates
                batch_coords = batch_coords.at[i, :, :n_points].set(coords)
                # Set validity mask
                batch_masks = batch_masks.at[i, :n_points].set(True)
            else:
                # Truncate if too large (should be rare with good buffer sizing)
                batch_coords = batch_coords.at[i, :, :].set(
                    coords[:, :target_buffer_size],
                )
                batch_masks = batch_masks.at[i, :].set(True)
                n_valid_list[-1] = target_buffer_size

        return batch_coords, batch_masks, n_valid_list

    @classmethod
    def optimize_compilation_cache(cls, max_cache_size: int = 50):
        """
        Optimize compilation cache based on usage patterns.

        Args:
            max_cache_size: Maximum number of compiled functions to cache
        """
        with cls._lock:
            if len(cls._compiled_functions) <= max_cache_size:
                return

            # Sort by usage patterns and keep most frequently used
            usage_scores = {}
            current_time = time.time()

            for (
                func_name,
                buffer_size,
            ), compiled_fn in cls._compiled_functions.items():
                if buffer_size in cls._usage_patterns:
                    pattern = cls._usage_patterns[buffer_size]
                    # Score based on usage frequency and recency
                    recency_score = 1.0 / max(1.0, current_time - pattern.last_used)
                    frequency_score = pattern.reuse_frequency
                    usage_scores[(func_name, buffer_size)] = (
                        recency_score * frequency_score
                    )
                else:
                    usage_scores[(func_name, buffer_size)] = 0.0

            # Keep top entries
            sorted_entries = sorted(
                usage_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            keys_to_keep = [key for key, score in sorted_entries[:max_cache_size]]

            # Clear cache and rebuild with top entries
            new_cache = {}
            for key in keys_to_keep:
                if key in cls._compiled_functions:
                    new_cache[key] = cls._compiled_functions[key]

            cls._compiled_functions = new_cache

            # Force garbage collection to free unused compiled functions
            gc.collect()

    @classmethod
    def get_buffer_usage_statistics(cls) -> Dict[str, Any]:
        """
        Get comprehensive buffer usage statistics.

        Returns:
            Dictionary with usage statistics and optimization recommendations
        """
        with cls._lock:
            current_time = time.time()

            # Analyze allocation patterns
            recent_allocations = [
                (ts, size, ctx)
                for ts, size, ctx in cls._allocation_history
                if current_time - ts < 3600  # Last hour
            ]

            size_distribution = defaultdict(int)
            context_distribution = defaultdict(int)

            for _, size, context in recent_allocations:
                # Find which buffer size this would use
                buffer_size = cls.determine_optimal_buffer_size(size, context)
                size_distribution[buffer_size] += 1
                context_distribution[context] += 1

            # Calculate buffer pool efficiency
            total_pools = len(cls._buffer_pools)
            total_pooled_buffers = sum(len(pool) for pool in cls._buffer_pools.values())

            # Generate optimization recommendations
            recommendations = []

            # Check for underutilized buffer sizes
            underutilized = [
                size
                for size, pattern in cls._usage_patterns.items()
                if pattern.reuse_frequency < 0.1 and pattern.usage_count > 5
            ]
            if underutilized:
                recommendations.append(
                    {
                        "type": "underutilized_buffers",
                        "message": f"Buffer sizes with low reuse: {underutilized[:5]}",
                        "suggestion": "Consider consolidating to fewer buffer sizes",
                    },
                )

            # Check for frequently used sizes
            hot_sizes = [
                size
                for size, pattern in cls._usage_patterns.items()
                if pattern.reuse_frequency > 0.3
            ]
            if hot_sizes:
                recommendations.append(
                    {
                        "type": "hot_buffer_sizes",
                        "message": f"Frequently used buffer sizes: {hot_sizes}",
                        "suggestion": "Consider precompiling functions for these sizes",
                    },
                )

            return {
                "total_allocations": len(cls._allocation_history),
                "recent_allocations": len(recent_allocations),
                "buffer_size_distribution": dict(size_distribution),
                "context_distribution": dict(context_distribution),
                "usage_patterns": {
                    size: {
                        "usage_count": pattern.usage_count,
                        "reuse_frequency": pattern.reuse_frequency,
                        "last_used_ago": current_time - pattern.last_used,
                    }
                    for size, pattern in cls._usage_patterns.items()
                },
                "buffer_pool_stats": {
                    "total_pools": total_pools,
                    "total_pooled_buffers": total_pooled_buffers,
                    "pool_sizes": {
                        str(key): len(pool) for key, pool in cls._buffer_pools.items()
                    },
                },
                "compilation_cache_stats": {
                    "cached_functions": len(cls._compiled_functions),
                    "cache_keys": list(cls._compiled_functions.keys()),
                },
                "optimization_recommendations": recommendations,
            }

    @classmethod
    def cleanup_unused_resources(cls, max_age_hours: float = 1.0):
        """
        Clean up unused resources to free memory.

        Args:
            max_age_hours: Maximum age in hours for keeping unused resources
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        with cls._lock:
            # Clean up old allocation history
            cls._allocation_history = [
                (ts, size, ctx)
                for ts, size, ctx in cls._allocation_history
                if current_time - ts < max_age_seconds * 24  # Keep 24 hours of history
            ]

            # Clean up unused buffer pools
            empty_pools = [
                key for key, pool in cls._buffer_pools.items() if len(pool) == 0
            ]
            for key in empty_pools:
                del cls._buffer_pools[key]

            # Limit buffer pool sizes
            for key, pool in cls._buffer_pools.items():
                if len(pool) > 10:  # Keep max 10 buffers per pool
                    cls._buffer_pools[key] = deque(list(pool)[:10])

            # Clean up old usage patterns
            old_patterns = [
                size
                for size, pattern in cls._usage_patterns.items()
                if current_time - pattern.last_used > max_age_seconds
            ]
            for size in old_patterns:
                del cls._usage_patterns[size]

        # Optimize compilation cache
        cls.optimize_compilation_cache()

        # Force garbage collection
        gc.collect()

    @classmethod
    def precompile_for_common_sizes(cls, common_sizes: Optional[List[int]] = None):
        """
        Precompile functions for commonly used buffer sizes.

        Args:
            common_sizes: List of buffer sizes to precompile for
        """
        if common_sizes is None:
            # Use most frequently used sizes from usage patterns
            with cls._lock:
                if cls._usage_patterns:
                    sorted_patterns = sorted(
                        cls._usage_patterns.items(),
                        key=lambda x: x[1].reuse_frequency,
                        reverse=True,
                    )
                    common_sizes = [size for size, _ in sorted_patterns[:10]]
                else:
                    common_sizes = [64, 128, 256, 512]  # Default common sizes

        print(f"Precompiling functions for buffer sizes: {common_sizes}")

        # Import here to avoid circular imports
        from .optimized_ops import OptimizedJaxAirfoilOps

        # Precompile common operations
        for size in common_sizes:
            try:
                # Create dummy data for compilation
                coords = jnp.zeros((2, size))
                mask = jnp.ones(size, dtype=bool)
                query_x = jnp.linspace(0, 1, 10)

                # Trigger compilation for thickness computation
                _ = OptimizedJaxAirfoilOps.compute_thickness_optimized(
                    coords,
                    coords,
                    size // 2,
                    size // 2,
                    query_x,
                    False,
                )

                print(f"  ✓ Precompiled for buffer size {size}")

            except Exception as e:
                print(f"  ✗ Failed to precompile for buffer size {size}: {e}")

        print("Precompilation complete.")


# Global instance for easy access
_optimized_buffer_manager = OptimizedAirfoilBufferManager()


# Convenience functions
def get_optimal_buffer_size(n_points: int, context: str = "general") -> int:
    """Get optimal buffer size for given points and context."""
    return OptimizedAirfoilBufferManager.determine_optimal_buffer_size(
        n_points,
        context,
    )


def get_buffer_usage_stats() -> Dict[str, Any]:
    """Get buffer usage statistics."""
    return OptimizedAirfoilBufferManager.get_buffer_usage_statistics()


def cleanup_buffer_resources():
    """Clean up unused buffer resources."""
    OptimizedAirfoilBufferManager.cleanup_unused_resources()


def precompile_common_buffer_sizes():
    """Precompile functions for common buffer sizes."""
    OptimizedAirfoilBufferManager.precompile_for_common_sizes()
