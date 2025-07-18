"""
Performance optimization utilities for JAX airfoil implementation.

This module provides comprehensive performance optimization features including:
- JIT compilation profiling and optimization
- Compilation caching strategies
- Memory-efficient buffer reuse mechanisms
- Gradient computation path optimization
- Performance benchmarking utilities
"""

import functools
import gc
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class CompilationStats:
    """Statistics for JIT compilation performance."""

    function_name: str
    compilation_time: float
    first_call_time: float
    subsequent_call_time: float
    static_args: Tuple[Any, ...]
    input_shapes: Tuple[Tuple[int, ...], ...]
    compilation_count: int = 0
    total_calls: int = 0

    def __post_init__(self):
        self.compilation_count = 1
        self.total_calls = 1


@dataclass
class MemoryStats:
    """Memory usage statistics for buffer management."""

    buffer_size: int
    active_buffers: int
    reused_buffers: int
    peak_memory_mb: float
    current_memory_mb: float
    allocation_count: int = 0
    deallocation_count: int = 0


class CompilationProfiler:
    """
    Profiler for JIT compilation performance analysis.

    Tracks compilation times, function call patterns, and provides
    optimization recommendations based on usage patterns.
    """

    def __init__(self):
        self._stats: Dict[str, CompilationStats] = {}
        self._lock = threading.Lock()

    def profile_function(self, func_name: str):
        """Decorator to profile JIT compilation performance."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Extract static arguments for tracking
                static_args = self._extract_static_args(func, args, kwargs)
                input_shapes = tuple(
                    arg.shape if hasattr(arg, "shape") else ()
                    for arg in args
                    if hasattr(arg, "shape")
                )

                start_time = time.perf_counter()

                # Check if this is first compilation
                cache_key = (func_name, static_args, input_shapes)
                is_first_call = cache_key not in self._stats

                if is_first_call:
                    # Time the compilation
                    result = func(*args, **kwargs)
                    compilation_time = time.perf_counter() - start_time

                    # Time a subsequent call to measure execution time
                    exec_start = time.perf_counter()
                    _ = func(*args, **kwargs)
                    exec_time = time.perf_counter() - exec_start

                    with self._lock:
                        self._stats[cache_key] = CompilationStats(
                            function_name=func_name,
                            compilation_time=compilation_time,
                            first_call_time=compilation_time,
                            subsequent_call_time=exec_time,
                            static_args=static_args,
                            input_shapes=input_shapes,
                        )
                else:
                    # Just time execution
                    result = func(*args, **kwargs)
                    exec_time = time.perf_counter() - start_time

                    with self._lock:
                        stats = self._stats[cache_key]
                        stats.total_calls += 1
                        # Update running average of execution time
                        stats.subsequent_call_time = (
                            stats.subsequent_call_time * (stats.total_calls - 1)
                            + exec_time
                        ) / stats.total_calls

                return result

            return wrapper

        return decorator

    def _extract_static_args(self, func, args, kwargs) -> Tuple[Any, ...]:
        """Extract static arguments from function call."""
        # Try to get static_argnums from the function
        if hasattr(func, "__wrapped__") and hasattr(func.__wrapped__, "static_argnums"):
            static_argnums = func.__wrapped__.static_argnums
            if static_argnums:
                return tuple(args[i] for i in static_argnums if i < len(args))
        return ()

    def get_compilation_report(self) -> Dict[str, Any]:
        """Generate comprehensive compilation performance report."""
        with self._lock:
            report = {
                "total_functions": len(self._stats),
                "functions": {},
                "optimization_recommendations": [],
            }

            total_compilation_time = 0
            slow_compilations = []

            for cache_key, stats in self._stats.items():
                func_name = stats.function_name
                if func_name not in report["functions"]:
                    report["functions"][func_name] = {
                        "variants": [],
                        "total_compilation_time": 0,
                        "avg_compilation_time": 0,
                        "total_calls": 0,
                    }

                func_report = report["functions"][func_name]
                func_report["variants"].append(
                    {
                        "static_args": stats.static_args,
                        "input_shapes": stats.input_shapes,
                        "compilation_time": stats.compilation_time,
                        "execution_time": stats.subsequent_call_time,
                        "compilation_count": stats.compilation_count,
                        "total_calls": stats.total_calls,
                        "efficiency_ratio": stats.subsequent_call_time
                        / max(stats.compilation_time, 1e-6),
                    },
                )

                func_report["total_compilation_time"] += stats.compilation_time
                func_report["total_calls"] += stats.total_calls
                total_compilation_time += stats.compilation_time

                # Track slow compilations
                if stats.compilation_time > 1.0:  # > 1 second
                    slow_compilations.append((func_name, stats.compilation_time))

            # Calculate averages
            for func_name, func_data in report["functions"].items():
                if func_data["variants"]:
                    func_data["avg_compilation_time"] = func_data[
                        "total_compilation_time"
                    ] / len(func_data["variants"])

            report["total_compilation_time"] = total_compilation_time

            # Generate optimization recommendations
            if slow_compilations:
                report["optimization_recommendations"].append(
                    {
                        "type": "slow_compilation",
                        "message": f"Functions with slow compilation (>1s): {slow_compilations[:5]}",
                        "suggestion": "Consider reducing complexity or splitting into smaller functions",
                    },
                )

            # Check for excessive recompilation
            excessive_variants = [
                (name, len(data["variants"]))
                for name, data in report["functions"].items()
                if len(data["variants"]) > 10
            ]
            if excessive_variants:
                report["optimization_recommendations"].append(
                    {
                        "type": "excessive_recompilation",
                        "message": f"Functions with many variants: {excessive_variants[:3]}",
                        "suggestion": "Review static argument usage and consider buffer size optimization",
                    },
                )

            return report


class CompilationCache:
    """
    Advanced compilation caching system for JAX functions.

    Provides intelligent caching strategies including:
    - LRU eviction for memory management
    - Precompilation for common patterns
    - Cache warming strategies
    """

    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self._cache: Dict[Any, Any] = {}
        self._access_order: List[Any] = []
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

    def get_or_compile(self, cache_key: Any, compile_fn: Callable) -> Any:
        """Get cached function or compile and cache it."""
        with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                self._hit_count += 1
                return self._cache[cache_key]

            # Cache miss - compile function
            self._miss_count += 1
            compiled_fn = compile_fn()

            # Add to cache
            self._cache[cache_key] = compiled_fn
            self._access_order.append(cache_key)

            # Evict if necessary
            if len(self._cache) > self.max_cache_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            return compiled_fn

    def precompile_common_patterns(
        self,
        patterns: List[Tuple[str, Callable, List[Any]]],
    ):
        """Precompile functions for common usage patterns."""
        for pattern_name, compile_fn, arg_sets in patterns:
            for args in arg_sets:
                cache_key = (pattern_name, args)
                if cache_key not in self._cache:
                    try:
                        self.get_or_compile(cache_key, lambda: compile_fn(*args))
                    except Exception as e:
                        print(
                            f"Warning: Failed to precompile {pattern_name} with args {args}: {e}",
                        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total_requests, 1)

            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }

    def clear_cache(self):
        """Clear the compilation cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class BufferPool:
    """
    Memory-efficient buffer reuse system.

    Manages a pool of pre-allocated buffers to reduce memory allocation
    overhead and improve performance for repeated operations.
    """

    def __init__(self):
        self._pools: Dict[Tuple[int, ...], List[jnp.ndarray]] = defaultdict(list)
        self._active_buffers: Dict[int, Tuple[int, ...]] = {}  # buffer_id -> shape
        self._lock = threading.Lock()
        self._allocation_count = 0
        self._reuse_count = 0
        self._peak_memory = 0
        self._current_memory = 0

    def get_buffer(self, shape: Tuple[int, ...], dtype=jnp.float32) -> jnp.ndarray:
        """Get a buffer from the pool or allocate a new one."""
        with self._lock:
            pool_key = (shape, dtype)

            if self._pools[pool_key]:
                # Reuse existing buffer
                buffer = self._pools[pool_key].pop()
                self._reuse_count += 1
                buffer_id = id(buffer)
                self._active_buffers[buffer_id] = shape
                return buffer
            else:
                # Allocate new buffer
                buffer = jnp.zeros(shape, dtype=dtype)
                self._allocation_count += 1
                buffer_id = id(buffer)
                self._active_buffers[buffer_id] = shape

                # Update memory tracking
                buffer_size = np.prod(shape) * np.dtype(dtype).itemsize
                self._current_memory += buffer_size
                self._peak_memory = max(self._peak_memory, self._current_memory)

                return buffer

    def return_buffer(self, buffer: jnp.ndarray):
        """Return a buffer to the pool for reuse."""
        with self._lock:
            buffer_id = id(buffer)
            if buffer_id in self._active_buffers:
                shape = self._active_buffers[buffer_id]
                pool_key = (shape, buffer.dtype)

                # Clear buffer data and return to pool
                buffer = jnp.zeros_like(buffer)
                self._pools[pool_key].append(buffer)
                del self._active_buffers[buffer_id]

    def get_memory_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self._pools.values())
            active_count = len(self._active_buffers)

            return MemoryStats(
                buffer_size=0,  # Will be set by caller
                active_buffers=active_count,
                reused_buffers=self._reuse_count,
                peak_memory_mb=self._peak_memory / (1024 * 1024),
                current_memory_mb=self._current_memory / (1024 * 1024),
                allocation_count=self._allocation_count,
            )

    def cleanup_unused_buffers(self, max_pool_size: int = 10):
        """Clean up unused buffers to free memory."""
        with self._lock:
            for pool_key, pool in self._pools.items():
                if len(pool) > max_pool_size:
                    # Keep only the most recently used buffers
                    excess_buffers = pool[max_pool_size:]
                    self._pools[pool_key] = pool[:max_pool_size]

                    # Update memory tracking
                    for buffer in excess_buffers:
                        buffer_size = np.prod(buffer.shape) * buffer.dtype.itemsize
                        self._current_memory -= buffer_size

            # Force garbage collection
            gc.collect()


class GradientOptimizer:
    """
    Optimization utilities for gradient computation paths.

    Provides strategies to optimize automatic differentiation performance
    including forward vs reverse mode selection and gradient checkpointing.
    """

    @staticmethod
    def select_grad_mode(n_inputs: int, n_outputs: int) -> str:
        """
        Select optimal gradient mode based on input/output dimensions.

        Args:
            n_inputs: Number of input parameters
            n_outputs: Number of output values

        Returns:
            Recommended gradient mode: 'forward', 'reverse', or 'mixed'
        """
        if n_inputs <= 4 and n_outputs > n_inputs * 2:
            return "forward"  # Forward mode efficient for few inputs
        elif n_outputs <= 4 and n_inputs > n_outputs * 2:
            return "reverse"  # Reverse mode efficient for few outputs
        else:
            return "mixed"  # Use mixed mode or default reverse

    @staticmethod
    def create_efficient_grad_fn(
        func: Callable,
        n_inputs: int,
        n_outputs: int,
    ) -> Callable:
        """Create optimized gradient function based on problem dimensions."""
        mode = GradientOptimizer.select_grad_mode(n_inputs, n_outputs)

        if mode == "forward":
            return jax.jacfwd(func)
        elif mode == "reverse":
            return jax.jacrev(func)
        else:
            # For mixed mode, use reverse by default (most common case)
            return jax.jacrev(func)

    @staticmethod
    def optimize_gradient_checkpointing(
        func: Callable,
        checkpoint_every: int = 5,
    ) -> Callable:
        """Apply gradient checkpointing to reduce memory usage."""
        return jax.checkpoint(func, prevent_cse=False)


# Global instances for easy access
_profiler = CompilationProfiler()
_cache = CompilationCache()
_buffer_pool = BufferPool()


# Convenience functions
def profile_jit(func_name: str):
    """Decorator to profile JIT compilation performance."""
    return _profiler.profile_function(func_name)


def get_compilation_report():
    """Get global compilation performance report."""
    return _profiler.get_compilation_report()


def get_buffer_from_pool(shape: Tuple[int, ...], dtype=jnp.float32):
    """Get buffer from global buffer pool."""
    return _buffer_pool.get_buffer(shape, dtype)


def return_buffer_to_pool(buffer: jnp.ndarray):
    """Return buffer to global buffer pool."""
    return _buffer_pool.return_buffer(buffer)


def get_memory_stats():
    """Get global memory usage statistics."""
    return _buffer_pool.get_memory_stats()


def cleanup_memory():
    """Clean up unused buffers and force garbage collection."""
    _buffer_pool.cleanup_unused_buffers()
    _cache.clear_cache()
    gc.collect()
