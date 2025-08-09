"""FailureManager for Monte Carlo failure analysis.

Provides the failure analysis engine for NetGraph. Supports parallel
processing, per-worker caching, and failure policy handling for workflow steps
and direct programmatic use.

Performance characteristics:
Time complexity: O(I × A / P), where I is iteration count, A is analysis cost,
and P is parallelism. Worker-local caching reduces repeated work when exclusion
sets repeat across iterations. Network serialization happens once per worker,
not per iteration.

Space complexity: O(V + E + I × R + C), where V and E are node and link counts,
R is result size per iteration, and C is cache size. The per-worker cache is
bounded and evicts in FIFO order after 1000 unique patterns.

Parallelism: For small iteration counts, serial execution avoids IPC overhead.
For larger workloads, parallel execution benefits from worker caching and CPU
utilization. Optimal parallelism is the number of CPU cores for analysis-bound
workloads.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Protocol, Set, TypeVar

from ngraph.algorithms.base import FlowPlacement
from ngraph.logging import get_logger
from ngraph.model.view import NetworkView
from ngraph.results.artifacts import FailurePolicySet

if TYPE_CHECKING:
    import cProfile

    from ngraph.model.network import Network

from ngraph.failure.policy import FailurePolicy

logger = get_logger(__name__)


def _create_cache_key(
    excluded_nodes: Set[str],
    excluded_links: Set[str],
    analysis_name: str,
    analysis_kwargs: Dict[str, Any],
) -> tuple:
    """Create cache key from exclusions, analysis name, and parameters.

    Args:
        excluded_nodes: Set of excluded node names.
        excluded_links: Set of excluded link IDs.
        analysis_name: Name of the analysis function.
        analysis_kwargs: Analysis function arguments.

    Returns:
        Tuple suitable for use as a cache key.
    """
    # Basic components that are always hashable
    base_key = (
        tuple(sorted(excluded_nodes)),
        tuple(sorted(excluded_links)),
        analysis_name,
    )

    # Normalize analysis_kwargs for hashing
    hashable_kwargs = []
    for key, value in sorted(analysis_kwargs.items()):
        try:
            # Try to hash the (key, value) pair directly
            _ = hash((key, value))
            hashable_kwargs.append((key, value))
        except TypeError:
            # For non-hashable values, use type name and a digest of a stable string
            value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
            hashable_kwargs.append((key, f"{type(value).__name__}_{value_hash}"))

    return base_key + (tuple(hashable_kwargs),)


def _auto_adjust_parallelism(parallelism: int, analysis_func: Any) -> int:
    """Adjust parallelism based on function characteristics.

    Args:
        parallelism: Requested parallelism level.
        analysis_func: Analysis function to check.

    Returns:
        Adjusted parallelism level.
    """
    # Check if function is defined in __main__ (notebook context)
    if hasattr(analysis_func, "__module__") and analysis_func.__module__ == "__main__":
        if parallelism > 1:
            logger.warning(
                "Function defined in notebook/script (__main__) detected. "
                "Forcing serial execution (parallelism=1) to avoid pickling issues. "
                "Consider moving analysis function to a separate module for parallel execution."
            )
        return 1

    return parallelism


# Global shared state for worker processes
_shared_network: "Network | None" = None
_analysis_cache: dict[tuple, Any] = {}

T = TypeVar("T")


class AnalysisFunction(Protocol):
    """Protocol for analysis functions used with FailureManager.

    Analysis functions should take a NetworkView and any additional
    keyword arguments, returning analysis results of any type.
    """

    def __call__(self, network_view: NetworkView, **kwargs) -> Any:
        """Execute analysis on network view with optional parameters."""
        ...


def _worker_init(network_pickle: bytes) -> None:
    """Initialize worker process with shared network and clear cache.

    Called once per worker process lifetime via ProcessPoolExecutor's
    initializer mechanism. Network is deserialized once per worker (not per task)
    to avoid repeated serialization overhead. Process boundaries provide
    isolation so no cross-contamination is possible.

    Args:
        network_pickle: Serialized Network object to deserialize and share.
    """
    global _shared_network, _analysis_cache

    # Each worker process has its own copy of globals (process isolation)
    _shared_network = pickle.loads(network_pickle)
    _analysis_cache.clear()

    worker_logger = get_logger(f"{__name__}.worker")
    worker_logger.debug(f"Worker {os.getpid()} initialized with network")


def _generic_worker(args: tuple[Any, ...]) -> tuple[Any, int, bool, set[str], set[str]]:
    """Execute analysis function with caching.

    Caches analysis results based on exclusion patterns and analysis parameters
    since many Monte Carlo iterations share the same exclusion sets.
    Analysis computation is deterministic for identical inputs, making caching safe.

    Args:
        args: Tuple containing (excluded_nodes, excluded_links, analysis_func,
              analysis_kwargs, iteration_index, is_baseline, analysis_name).

    Returns:
        Tuple of (analysis_result, iteration_index, is_baseline,
                 excluded_nodes, excluded_links).
    """
    global _shared_network, _analysis_cache

    if _shared_network is None:
        raise RuntimeError("Worker not initialized with network data")

    worker_logger = get_logger(f"{__name__}.worker")

    (
        excluded_nodes,
        excluded_links,
        analysis_func,
        analysis_kwargs,
        iteration_index,
        is_baseline,
        analysis_name,
    ) = args

    # Optional per-worker profiling for performance analysis
    profile_dir_env = os.getenv("NGRAPH_PROFILE_DIR")
    collect_profile: bool = bool(profile_dir_env)

    profiler: "cProfile.Profile | None" = None
    if collect_profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    worker_pid = os.getpid()
    worker_logger.debug(
        f"Worker {worker_pid} starting: iteration={iteration_index}, "
        f"excluded_nodes={len(excluded_nodes)}, excluded_links={len(excluded_links)}"
    )

    # Create cache key from all parameters affecting analysis computation
    # Sorting ensures consistent keys for same sets regardless of iteration order
    cache_key = _create_cache_key(
        excluded_nodes, excluded_links, analysis_name, analysis_kwargs
    )

    # Check cache first since analysis computation is deterministic
    if cache_key in _analysis_cache:
        worker_logger.debug(f"Worker {worker_pid} using cached analysis results")
        result = _analysis_cache[cache_key]
    else:
        worker_logger.debug(f"Worker {worker_pid} computing analysis (cache miss)")

        # Use NetworkView for exclusion without copying network
        network_view = NetworkView.from_excluded_sets(
            _shared_network,
            excluded_nodes=excluded_nodes,
            excluded_links=excluded_links,
        )
        worker_logger.debug(f"Worker {worker_pid} created NetworkView")

        # Execute analysis function
        worker_logger.debug(f"Worker {worker_pid} executing {analysis_name}")
        result = analysis_func(network_view, **analysis_kwargs)

        # Cache results for future computations
        _analysis_cache[cache_key] = result

        # Bound cache size to prevent memory exhaustion (FIFO eviction)
        if len(_analysis_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            for _ in range(100):
                _analysis_cache.pop(next(iter(_analysis_cache)))

        worker_logger.debug(f"Worker {worker_pid} completed analysis")

    # Dump profile if enabled (for performance analysis)
    if profiler is not None:
        profiler.disable()
        try:
            import pstats
            import uuid
            from pathlib import Path

            profile_dir = Path(profile_dir_env) if profile_dir_env else None
            if profile_dir is not None:
                profile_dir.mkdir(parents=True, exist_ok=True)
                unique_id = uuid.uuid4().hex[:8]
                profile_path = (
                    profile_dir
                    / f"{analysis_name}_worker_{worker_pid}_{unique_id}.pstats"
                )
                pstats.Stats(profiler).dump_stats(profile_path)
                worker_logger.debug("Saved worker profile to %s", profile_path.name)
        except Exception as exc:  # pragma: no cover
            worker_logger.warning(
                "Failed to save worker profile: %s: %s", type(exc).__name__, exc
            )

    return (result, iteration_index, is_baseline, excluded_nodes, excluded_links)


class FailureManager:
    """Failure analysis engine with Monte Carlo capabilities.

    This is the component for failure analysis in NetGraph.
    Provides parallel processing, worker caching, and failure
    policy handling for workflow steps and direct notebook usage.

    The FailureManager can execute any analysis function that takes a NetworkView
    and returns results, making it generic for different types of
    failure analysis (capacity, traffic, connectivity, etc.).

    Attributes:
        network: The underlying network (not modified during analysis).
        failure_policy_set: Set of named failure policies.
        policy_name: Name of specific failure policy to use.
    """

    def __init__(
        self,
        network: "Network",
        failure_policy_set: FailurePolicySet,
        policy_name: str | None = None,
    ) -> None:
        """Initialize FailureManager.

        Args:
            network: Network to analyze (read-only, not modified).
            failure_policy_set: Set of named failure policies.
            policy_name: Name of specific policy to use. If None, no failure policy is applied.
        """
        self.network = network
        self.failure_policy_set = failure_policy_set
        self.policy_name = policy_name

    def get_failure_policy(self) -> "FailurePolicy | None":
        """Get failure policy for analysis.

        Returns:
            FailurePolicy instance or None if no policy should be applied.

        Raises:
            ValueError: If named policy is not found in failure_policy_set.
        """
        if self.policy_name is not None:
            try:
                return self.failure_policy_set.get_policy(self.policy_name)
            except KeyError:
                raise ValueError(
                    f"Failure policy '{self.policy_name}' not found in scenario"
                ) from None
        else:
            return None

    def compute_exclusions(
        self,
        policy: "FailurePolicy | None" = None,
        seed_offset: int | None = None,
    ) -> tuple[set[str], set[str]]:
        """Compute set of nodes and links to exclude for a failure iteration.

        Applies failure policy logic and returns exclusion sets. This is
        equivalent to applying failures to the network and then filtering, but
        with lower overhead since exclusion sets are typically small.

        Args:
            policy: Failure policy to apply. If None, uses instance policy.
            seed_offset: Optional seed for deterministic failures.

        Returns:
            Tuple of (excluded_nodes, excluded_links) containing entity IDs to exclude.
        """
        if policy is None:
            policy = self.get_failure_policy()

        excluded_nodes = set()
        excluded_links = set()

        if policy is None:
            return excluded_nodes, excluded_links

        # Build merged views of nodes and links including top-level fields required by
        # policy matching and risk-group expansion. This ensures attributes like
        # 'risk_groups' are available to the policy engine.
        def _merge_node_attrs() -> dict[str, dict[str, Any]]:
            merged: dict[str, dict[str, Any]] = {}
            for node_name, node in self.network.nodes.items():
                attrs: dict[str, Any] = {
                    "name": node.name,
                    "disabled": node.disabled,
                    "risk_groups": node.risk_groups,
                }
                # Top-level fields take precedence over attrs on conflict
                attrs.update({k: v for k, v in node.attrs.items() if k not in attrs})
                merged[node_name] = attrs
            return merged

        def _merge_link_attrs() -> dict[str, dict[str, Any]]:
            merged: dict[str, dict[str, Any]] = {}
            for link_id, link in self.network.links.items():
                attrs: dict[str, Any] = {
                    "id": link_id,
                    "source": link.source,
                    "target": link.target,
                    "capacity": link.capacity,
                    "cost": link.cost,
                    "disabled": link.disabled,
                    "risk_groups": link.risk_groups,
                }
                attrs.update({k: v for k, v in link.attrs.items() if k not in attrs})
                merged[link_id] = attrs
            return merged

        node_map = _merge_node_attrs()
        link_map = _merge_link_attrs()

        # Apply failure policy with optional deterministic seed override
        failed_ids = policy.apply_failures(
            node_map, link_map, self.network.risk_groups, seed=seed_offset
        )

        # Separate entity types for NetworkView creation
        for f_id in failed_ids:
            if f_id in self.network.nodes:
                excluded_nodes.add(f_id)
            elif f_id in self.network.links:
                excluded_links.add(f_id)
            elif f_id in self.network.risk_groups:
                # Recursively expand risk groups
                risk_group = self.network.risk_groups[f_id]
                to_check = [risk_group]
                while to_check:
                    grp = to_check.pop()
                    # Add all nodes/links in this risk group
                    for node_name, node in self.network.nodes.items():
                        if grp.name in node.risk_groups:
                            excluded_nodes.add(node_name)
                    for link_id, link in self.network.links.items():
                        if grp.name in link.risk_groups:
                            excluded_links.add(link_id)
                    # Check children recursively
                    to_check.extend(grp.children)

        return excluded_nodes, excluded_links

    def create_network_view(
        self,
        excluded_nodes: set[str] | None = None,
        excluded_links: set[str] | None = None,
    ) -> NetworkView:
        """Create NetworkView with specified exclusions.

        Args:
            excluded_nodes: Set of node IDs to exclude. Empty set if None.
            excluded_links: Set of link IDs to exclude. Empty set if None.

        Returns:
            NetworkView with exclusions applied, or original network if no exclusions.
        """
        if not excluded_nodes and not excluded_links:
            # Return NetworkView with no exclusions instead of raw Network
            return NetworkView.from_excluded_sets(
                self.network,
                excluded_nodes=set(),
                excluded_links=set(),
            )

        return NetworkView.from_excluded_sets(
            self.network,
            excluded_nodes=excluded_nodes or set(),
            excluded_links=excluded_links or set(),
        )

    def run_monte_carlo_analysis(
        self,
        analysis_func: AnalysisFunction,
        iterations: int = 1,
        parallelism: int = 1,
        baseline: bool = False,
        seed: int | None = None,
        store_failure_patterns: bool = False,
        **analysis_kwargs,
    ) -> dict[str, Any]:
        """Run Monte Carlo failure analysis with any analysis function.

        This is the main method for executing failure analysis. Handles
        parallel processing, worker caching, and failure policy
        application, while allowing flexibility in the analysis function.

        Args:
            analysis_func: Function that takes (network_view, **kwargs) and returns results.
                          Must be serializable for parallel execution.
            iterations: Number of Monte Carlo iterations to run.
            parallelism: Number of parallel worker processes to use.
            baseline: If True, first iteration runs without failures as baseline.
            seed: Optional seed for reproducible results across runs.
            store_failure_patterns: If True, store detailed failure patterns in results.
            **analysis_kwargs: Additional arguments passed to analysis_func.

        Returns:
            Dictionary containing:
            - 'results': List of results from each iteration
            - 'failure_patterns': List of failure pattern details (if store_failure_patterns=True)
            - 'metadata': Execution metadata (iterations, timing, etc.)

        Raises:
            ValueError: If iterations > 1 without a failure policy and baseline=False.
        """
        policy = self.get_failure_policy()

        # Validate iterations parameter based on failure policy
        if (policy is None or not policy.rules) and iterations > 1 and not baseline:
            raise ValueError(
                f"iterations={iterations} has no effect without a failure policy. "
                "Without failures, all iterations produce the same results. "
                "Either set iterations=1, provide a failure_policy with rules, or set baseline=True."
            )

        if baseline and iterations < 2:
            raise ValueError(
                "baseline=True requires iterations >= 2 "
                "(first iteration is baseline, remaining are with failures)"
            )

        # Auto-adjust parallelism based on function characteristics
        parallelism = _auto_adjust_parallelism(parallelism, analysis_func)

        # Determine actual number of iterations to run
        if policy is None or not policy.rules:
            mc_iters = 1  # Baseline only, no failures
        else:
            mc_iters = iterations

        logger.info(f"Running {mc_iters} Monte-Carlo iterations")

        # Get function name safely (Protocol doesn't guarantee __name__)
        func_name = getattr(analysis_func, "__name__", "analysis_function")
        logger.debug(
            f"Analysis parameters: function={func_name}, "
            f"parallelism={parallelism}, baseline={baseline}, policy={self.policy_name}"
        )

        # Pre-compute worker arguments for all iterations
        logger.debug("Pre-computing failure exclusions for all iterations")
        pre_compute_start = time.time()

        worker_args = []
        for i in range(mc_iters):
            seed_offset = None
            if seed is not None:
                seed_offset = seed + i

            # First iteration is baseline if baseline=True (no failures)
            is_baseline = baseline and i == 0

            if is_baseline:
                # For baseline iteration, use empty exclusion sets
                excluded_nodes, excluded_links = set(), set()
            else:
                # Pre-compute exclusions for this iteration
                excluded_nodes, excluded_links = self.compute_exclusions(
                    policy, seed_offset
                )

            # Create worker arguments
            worker_args.append(
                (
                    excluded_nodes,
                    excluded_links,
                    analysis_func,
                    analysis_kwargs,
                    i,  # iteration_index
                    is_baseline,
                    func_name,
                )
            )

        pre_compute_time = time.time() - pre_compute_start
        logger.debug(
            f"Pre-computed {len(worker_args)} exclusion sets in {pre_compute_time:.2f}s"
        )

        # Determine if we should run in parallel
        use_parallel = parallelism > 1 and mc_iters > 1

        start_time = time.time()

        if use_parallel:
            results, failure_patterns = self._run_parallel(
                worker_args, mc_iters, store_failure_patterns, parallelism
            )
        else:
            results, failure_patterns = self._run_serial(
                worker_args, store_failure_patterns
            )

        elapsed_time = time.time() - start_time

        return {
            "results": results,
            "failure_patterns": failure_patterns if store_failure_patterns else [],
            "metadata": {
                "iterations": mc_iters,
                "parallelism": parallelism,
                "baseline": baseline,
                "analysis_function": func_name,
                "policy_name": self.policy_name,
                "execution_time": elapsed_time,
                "unique_patterns": len(
                    set(
                        (tuple(sorted(args[0])), tuple(sorted(args[1])))
                        for args in worker_args
                    )
                ),
            },
        }

    def _run_parallel(
        self,
        worker_args: list[tuple],
        mc_iters: int,
        store_failure_patterns: bool,
        parallelism: int,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run analysis in parallel using shared network approach.

        Network is serialized once in the main process and deserialized once per
        worker via the initializer, avoiding repeated serialization overhead.
        Each worker receives only small exclusion sets instead of modified network
        copies, reducing IPC overhead.

        Args:
            worker_args: Pre-computed worker arguments for all iterations.
            mc_iters: Number of iterations to run.
            store_failure_patterns: Whether to collect failure pattern details.
            parallelism: Number of parallel worker processes to use.

        Returns:
            Tuple of (results_list, failure_patterns_list).
        """
        workers = min(parallelism, mc_iters)
        logger.info(
            f"Running parallel analysis with {workers} workers for {mc_iters} iterations"
        )

        # Serialize network once for all workers
        network_pickle = pickle.dumps(self.network)
        logger.debug(f"Serialized network once: {len(network_pickle)} bytes")

        # Calculate optimal chunksize to minimize IPC overhead
        chunksize = max(1, mc_iters // (workers * 4))
        logger.debug(f"Using chunksize={chunksize} for parallel execution")

        start_time = time.time()
        completed_tasks = 0
        results = []
        failure_patterns = []

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(network_pickle,),
        ) as pool:
            logger.debug(
                f"ProcessPoolExecutor created with {workers} workers and shared network"
            )
            logger.info(f"Starting parallel execution of {mc_iters} iterations")

            try:
                for (
                    result,
                    iteration_index,
                    is_baseline,
                    excluded_nodes,
                    excluded_links,
                ) in pool.map(_generic_worker, worker_args, chunksize=chunksize):
                    completed_tasks += 1

                    # Collect results
                    results.append(result)

                    # Add failure pattern if requested
                    if store_failure_patterns:
                        failure_patterns.append(
                            {
                                "iteration_index": iteration_index,
                                "is_baseline": is_baseline,
                                "excluded_nodes": list(excluded_nodes),
                                "excluded_links": list(excluded_links),
                            }
                        )

                    # Progress logging
                    if completed_tasks % max(1, mc_iters // 10) == 0:
                        logger.info(
                            f"Parallel analysis progress: {completed_tasks}/{mc_iters} tasks completed"
                        )

            except Exception as e:
                logger.error(
                    f"Error during parallel execution: {type(e).__name__}: {e}"
                )
                logger.debug(f"Failed after {completed_tasks} completed tasks")
                raise

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel analysis completed in {elapsed_time:.2f} seconds")
        logger.debug(
            f"Average time per iteration: {elapsed_time / mc_iters:.3f} seconds"
        )

        # Log exclusion pattern diversity for cache efficiency analysis
        unique_exclusions = set()
        for args in worker_args:
            excluded_nodes, excluded_links = args[0], args[1]
            exclusion_key = (
                tuple(sorted(excluded_nodes)),
                tuple(sorted(excluded_links)),
            )
            unique_exclusions.add(exclusion_key)

        logger.info(
            f"Generated {len(unique_exclusions)} unique exclusion patterns from {mc_iters} iterations"
        )
        cache_efficiency = (mc_iters - len(unique_exclusions)) / mc_iters * 100
        logger.debug(
            f"Potential cache efficiency: {cache_efficiency:.1f}% (worker processes benefit from caching)"
        )

        return results, failure_patterns

    def _run_serial(
        self,
        worker_args: list[tuple],
        store_failure_patterns: bool,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Run analysis serially for single process execution.

        Args:
            worker_args: Pre-computed worker arguments for all iterations.
            store_failure_patterns: Whether to collect failure pattern details.

        Returns:
            Tuple of (results_list, failure_patterns_list).
        """
        logger.info("Running serial analysis")
        start_time = time.time()

        # For serial execution, we need to initialize the global network
        global _shared_network
        _shared_network = self.network

        results = []
        failure_patterns = []

        try:
            for i, args in enumerate(worker_args):
                iter_start = time.time()

                is_baseline = len(args) > 5 and args[5]  # is_baseline flag
                baseline_msg = " (baseline)" if is_baseline else ""
                logger.debug(
                    f"Serial iteration {i + 1}/{len(worker_args)}{baseline_msg}"
                )

                (
                    result,
                    iteration_index,
                    is_baseline,
                    excluded_nodes,
                    excluded_links,
                ) = _generic_worker(args)

                # Collect results
                results.append(result)

                # Add failure pattern if requested
                if store_failure_patterns:
                    failure_patterns.append(
                        {
                            "iteration_index": iteration_index,
                            "is_baseline": is_baseline,
                            "excluded_nodes": list(excluded_nodes),
                            "excluded_links": list(excluded_links),
                        }
                    )

                iter_time = time.time() - iter_start
                if len(worker_args) <= 10:
                    logger.debug(
                        f"Serial iteration {i + 1} completed in {iter_time:.3f} seconds"
                    )

                if (
                    len(worker_args) > 1
                    and (i + 1) % max(1, len(worker_args) // 10) == 0
                ):
                    logger.info(
                        f"Serial analysis progress: {i + 1}/{len(worker_args)} iterations completed"
                    )
        finally:
            # Clean up global network reference
            _shared_network = None

        elapsed_time = time.time() - start_time
        logger.info(f"Serial analysis completed in {elapsed_time:.2f} seconds")
        if len(worker_args) > 1:
            logger.debug(
                f"Average time per iteration: {elapsed_time / len(worker_args):.3f} seconds"
            )
        logger.info(
            f"Analysis cache contains {len(_analysis_cache)} unique patterns after serial analysis"
        )

        return results, failure_patterns

    def run_single_failure_scenario(
        self, analysis_func: AnalysisFunction, **kwargs
    ) -> Any:
        """Run a single failure scenario for convenience.

        This is a convenience method for running a single iteration, useful for
        quick analysis or debugging. For full Monte Carlo analysis, use
        run_monte_carlo_analysis().

        Args:
            analysis_func: Function that takes (network_view, **kwargs) and returns results.
            **kwargs: Additional arguments passed to analysis_func.

        Returns:
            Result from the analysis function.
        """
        result = self.run_monte_carlo_analysis(
            analysis_func=analysis_func, iterations=1, parallelism=1, **kwargs
        )
        return result["results"][0]

    # Convenience methods for common analysis patterns

    def run_max_flow_monte_carlo(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        iterations: int = 100,
        parallelism: int = 1,
        shortest_path: bool = False,
        flow_placement: FlowPlacement | str = FlowPlacement.PROPORTIONAL,
        baseline: bool = False,
        seed: int | None = None,
        store_failure_patterns: bool = False,
        include_flow_summary: bool = False,
        **kwargs,
    ) -> Any:  # Will be CapacityEnvelopeResults when imports are enabled
        """Analyze maximum flow capacity envelopes between node groups under failures.

        Computes statistical distributions (envelopes) of maximum flow capacity between
        source and sink node groups across Monte Carlo failure scenarios. Results include
        frequency-based capacity envelopes and optional failure pattern analysis.

        Args:
            source_path: Regex pattern for source node groups.
            sink_path: Regex pattern for sink node groups.
            mode: "combine" (aggregate) or "pairwise" (individual flows).
            iterations: Number of failure scenarios to simulate.
            parallelism: Number of parallel workers (auto-adjusted if needed).
            shortest_path: Whether to use shortest paths only.
            flow_placement: Flow placement strategy.
            baseline: Whether to include baseline (no failures) iteration.
            seed: Optional seed for reproducible results.
            store_failure_patterns: Whether to store failure patterns in results.
            include_flow_summary: Whether to collect detailed flow summary data.

        Returns:
            CapacityEnvelopeResults object with envelope statistics and analysis methods.
        """
        from ngraph.monte_carlo.functions import max_flow_analysis
        from ngraph.monte_carlo.results import CapacityEnvelopeResults

        # Convert string flow_placement to enum if needed
        if isinstance(flow_placement, str):
            try:
                flow_placement = FlowPlacement[flow_placement.upper()]
            except KeyError:
                valid_values = ", ".join([e.name for e in FlowPlacement])
                raise ValueError(
                    f"Invalid flow_placement '{flow_placement}'. "
                    f"Valid values are: {valid_values}"
                ) from None

        # Run Monte Carlo analysis
        raw_results = self.run_monte_carlo_analysis(
            analysis_func=max_flow_analysis,
            iterations=iterations,
            parallelism=parallelism,
            baseline=baseline,
            seed=seed,
            store_failure_patterns=store_failure_patterns,
            source_regex=source_path,
            sink_regex=sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=flow_placement,
            include_flow_summary=include_flow_summary,
            **kwargs,
        )

        # Process results with support for flow summary data
        if include_flow_summary:
            samples, flow_summaries = self._process_results_with_summaries(
                raw_results["results"]
            )
            envelopes = self._build_capacity_envelopes_with_summaries(
                samples, flow_summaries, source_path, sink_path, mode
            )
        else:
            samples = self._process_results_to_samples(raw_results["results"])
            envelopes = self._build_capacity_envelopes(
                samples, source_path, sink_path, mode
            )

        # Process failure patterns if requested
        failure_patterns = {}
        if store_failure_patterns and raw_results["failure_patterns"]:
            failure_patterns = self._build_failure_pattern_results(
                raw_results["failure_patterns"], samples
            )

        return CapacityEnvelopeResults(
            envelopes=envelopes,
            failure_patterns=failure_patterns,
            source_pattern=source_path,
            sink_pattern=sink_path,
            mode=mode,
            iterations=iterations,
            metadata=raw_results["metadata"],
        )

    def _process_results_to_samples(
        self, results: list[list[tuple[str, str, float]]]
    ) -> dict[tuple[str, str], list[float]]:
        """Convert raw results from FailureManager to samples dictionary.

        Args:
            results: List of results from each iteration, where each result
                    is a list of (source, sink, capacity) tuples.

        Returns:
            Dictionary mapping (source, sink) to list of capacity values.
        """
        from collections import defaultdict

        samples = defaultdict(list)

        for flow_results in results:
            for src, dst, capacity in flow_results:
                samples[(src, dst)].append(capacity)

        logger.debug(f"Processed samples for {len(samples)} flow pairs")
        return samples

    def _build_capacity_envelopes(
        self,
        samples: dict[tuple[str, str], list[float]],
        source_pattern: str,
        sink_pattern: str,
        mode: str,
    ) -> dict[str, Any]:
        """Build CapacityEnvelope objects from collected samples.

        Args:
            samples: Dictionary mapping (src_label, dst_label) to capacity values.
            source_pattern: Source node regex pattern
            sink_pattern: Sink node regex pattern
            mode: Flow analysis mode

        Returns:
            Dictionary mapping flow keys to CapacityEnvelope objects.
        """
        from ngraph.results.artifacts import CapacityEnvelope

        envelopes = {}

        for (src_label, dst_label), capacity_values in samples.items():
            if not capacity_values:
                logger.warning(
                    f"No capacity values found for flow {src_label}->{dst_label}"
                )
                continue

            # Use flow key as the result key
            flow_key = f"{src_label}->{dst_label}"

            # Create frequency-based envelope
            envelope = CapacityEnvelope.from_values(
                source_pattern=source_pattern,
                sink_pattern=sink_pattern,
                mode=mode,
                values=capacity_values,
            )
            envelopes[flow_key] = envelope

            logger.debug(
                f"Created envelope for {flow_key}: {envelope.total_samples} samples, "
                f"min={envelope.min_capacity:.2f}, max={envelope.max_capacity:.2f}, "
                f"mean={envelope.mean_capacity:.2f}"
            )

        return envelopes

    def _process_results_with_summaries(
        self, results: list[list[tuple]]
    ) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], list[Any]]]:
        """Convert raw results with FlowSummary data to samples and summaries dictionaries.

        Args:
            results: List of results from each iteration, where each result
                    is a list of (source, sink, capacity, flow_summary) tuples.

        Returns:
            Tuple of:
            - Dictionary mapping (source, sink) to list of capacity values
            - Dictionary mapping (source, sink) to list of FlowSummary objects
        """
        from collections import defaultdict

        samples = defaultdict(list)
        flow_summaries = defaultdict(list)

        for flow_results in results:
            for result_tuple in flow_results:
                if len(result_tuple) == 4:
                    # Format: (src, dst, capacity, flow_summary)
                    src, dst, capacity, summary = result_tuple
                    samples[(src, dst)].append(capacity)
                    flow_summaries[(src, dst)].append(summary)
                elif len(result_tuple) == 3:
                    # Fallback for backwards compatibility: (src, dst, capacity)
                    src, dst, capacity = result_tuple
                    samples[(src, dst)].append(capacity)
                    # No summary available, append None
                    flow_summaries[(src, dst)].append(None)
                else:
                    logger.warning(f"Unexpected result tuple format: {result_tuple}")

        logger.debug(f"Processed samples and summaries for {len(samples)} flow pairs")
        return samples, flow_summaries

    def _build_capacity_envelopes_with_summaries(
        self,
        samples: dict[tuple[str, str], list[float]],
        flow_summaries: dict[tuple[str, str], list[Any]],
        source_pattern: str,
        sink_pattern: str,
        mode: str,
    ) -> dict[str, Any]:
        """Build CapacityEnvelope objects from collected samples and flow summaries.

        Args:
            samples: Dictionary mapping (src_label, dst_label) to capacity values.
            flow_summaries: Dictionary mapping (src_label, dst_label) to FlowSummary objects.
            source_pattern: Source node regex pattern
            sink_pattern: Sink node regex pattern
            mode: Flow analysis mode

        Returns:
            Dictionary mapping flow keys to CapacityEnvelope objects with flow summary data.
        """
        from ngraph.results.artifacts import CapacityEnvelope

        envelopes = {}

        for (src_label, dst_label), capacity_values in samples.items():
            if not capacity_values:
                logger.warning(
                    f"No capacity values found for flow {src_label}->{dst_label}"
                )
                continue

            # Use flow key as the result key
            flow_key = f"{src_label}->{dst_label}"

            # Get corresponding flow summaries
            summaries = flow_summaries.get((src_label, dst_label), [])

            # Extract cost distribution data from summaries
            cost_distributions = []
            for summary in summaries:
                if summary is not None and hasattr(summary, "cost_distribution"):
                    cost_distributions.append(summary.cost_distribution)

            # Create frequency-based envelope with flow summary statistics
            envelope = CapacityEnvelope.from_values(
                source_pattern=source_pattern,
                sink_pattern=sink_pattern,
                mode=mode,
                values=capacity_values,
                flow_summaries=summaries,
            )

            envelopes[flow_key] = envelope

            logger.debug(
                f"Created envelope for {flow_key}: {envelope.total_samples} samples, "
                f"min={envelope.min_capacity:.2f}, max={envelope.max_capacity:.2f}, "
                f"mean={envelope.mean_capacity:.2f}, flow_summaries={len(summaries)}, "
                f"cost_levels={len(envelope.flow_summary_stats.get('cost_distribution_stats', {}))}"
            )

        return envelopes

    def _build_failure_pattern_results(
        self,
        failure_patterns: list[dict[str, Any]],
        samples: dict[tuple[str, str], list[float]],
    ) -> dict[str, Any]:
        """Build failure pattern results from collected patterns and samples.

        Args:
            failure_patterns: List of failure pattern details from FailureManager.
            samples: Sample data for building capacity matrices.

        Returns:
            Dictionary mapping pattern keys to FailurePatternResult objects.
        """
        import json

        from ngraph.results.artifacts import FailurePatternResult

        pattern_map = {}

        for pattern in failure_patterns:
            # Create pattern key from exclusions
            key = json.dumps(
                {
                    "excluded_nodes": pattern["excluded_nodes"],
                    "excluded_links": pattern["excluded_links"],
                },
                sort_keys=True,
            )

            if key not in pattern_map:
                # Get capacity matrix for this pattern
                capacity_matrix = {}
                pattern_iter = pattern["iteration_index"]

                for (src, dst), values in samples.items():
                    if pattern_iter < len(values):
                        flow_key = f"{src}->{dst}"
                        capacity_matrix[flow_key] = values[pattern_iter]

                pattern_map[key] = FailurePatternResult(
                    excluded_nodes=pattern["excluded_nodes"],
                    excluded_links=pattern["excluded_links"],
                    capacity_matrix=capacity_matrix,
                    count=0,
                    is_baseline=pattern["is_baseline"],
                )

            pattern_map[key].count += 1

        # Return FailurePatternResult objects directly
        return {result.pattern_key: result for result in pattern_map.values()}

    def _build_demand_placement_failure_patterns(
        self,
        failure_patterns: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build failure pattern results for demand placement analysis.

        Args:
            failure_patterns: List of failure pattern details from FailureManager.
            results: List of placement results for building pattern analysis.

        Returns:
            Dictionary mapping pattern keys to demand placement pattern results.
        """
        import json

        pattern_map = {}

        for i, pattern in enumerate(failure_patterns):
            # Create pattern key from exclusions
            key = json.dumps(
                {
                    "excluded_nodes": pattern["excluded_nodes"],
                    "excluded_links": pattern["excluded_links"],
                },
                sort_keys=True,
            )

            if key not in pattern_map:
                # Get placement result for this pattern
                placement_result = results[i] if i < len(results) else {}

                pattern_map[key] = {
                    "excluded_nodes": pattern["excluded_nodes"],
                    "excluded_links": pattern["excluded_links"],
                    "placement_result": placement_result,
                    "count": 0,
                    "is_baseline": pattern["is_baseline"],
                }

            pattern_map[key]["count"] += 1

        return pattern_map

    def _process_sensitivity_results(
        self, results: list[dict[str, dict[str, float]]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Process sensitivity results to aggregate component impact scores.

        Args:
            results: List of sensitivity results from each iteration.

        Returns:
            Dictionary mapping flow keys to component impact aggregations.
        """
        from collections import defaultdict

        # Aggregate component scores across all iterations
        flow_aggregates = defaultdict(lambda: defaultdict(list))

        for result in results:
            for flow_key, components in result.items():
                for component_key, score in components.items():
                    flow_aggregates[flow_key][component_key].append(score)

        # Calculate statistics for each component
        processed_scores = {}
        for flow_key, components in flow_aggregates.items():
            flow_stats = {}
            for component_key, scores in components.items():
                if scores:
                    flow_stats[component_key] = {
                        "mean": sum(scores) / len(scores),
                        "max": max(scores),
                        "min": min(scores),
                        "count": len(scores),
                    }
            processed_scores[flow_key] = flow_stats

        logger.debug(
            f"Processed sensitivity scores for {len(processed_scores)} flow pairs"
        )
        return processed_scores

    def _build_sensitivity_failure_patterns(
        self,
        failure_patterns: list[dict[str, Any]],
        results: list[dict[str, dict[str, float]]],
    ) -> dict[str, Any]:
        """Build failure pattern results for sensitivity analysis.

        Args:
            failure_patterns: List of failure pattern details from FailureManager.
            results: List of sensitivity results for building pattern analysis.

        Returns:
            Dictionary mapping pattern keys to sensitivity pattern results.
        """
        import json

        pattern_map = {}

        for i, pattern in enumerate(failure_patterns):
            # Create pattern key from exclusions
            key = json.dumps(
                {
                    "excluded_nodes": pattern["excluded_nodes"],
                    "excluded_links": pattern["excluded_links"],
                },
                sort_keys=True,
            )

            if key not in pattern_map:
                # Get sensitivity result for this pattern
                sensitivity_result = results[i] if i < len(results) else {}

                pattern_map[key] = {
                    "excluded_nodes": pattern["excluded_nodes"],
                    "excluded_links": pattern["excluded_links"],
                    "sensitivity_result": sensitivity_result,
                    "count": 0,
                    "is_baseline": pattern["is_baseline"],
                }

            pattern_map[key]["count"] += 1

        return pattern_map

    def run_demand_placement_monte_carlo(
        self,
        demands_config: list[dict[str, Any]]
        | Any,  # List of demand configs or TrafficMatrixSet
        iterations: int = 100,
        parallelism: int = 1,
        placement_rounds: int = 50,
        baseline: bool = False,
        seed: int | None = None,
        store_failure_patterns: bool = False,
        **kwargs,
    ) -> Any:  # Will be DemandPlacementResults when imports are enabled
        """Analyze traffic demand placement success under failures.

        Attempts to place traffic demands on the network across
        Monte Carlo failure scenarios and measures success rates.

        Args:
            demands_config: List of demand configs or TrafficMatrixSet object.
            iterations: Number of failure scenarios to simulate.
            parallelism: Number of parallel workers (auto-adjusted if needed).
            placement_rounds: Optimization rounds for demand placement.
            baseline: Whether to include baseline (no failures) iteration.
            seed: Optional seed for reproducible results.
            store_failure_patterns: Whether to store failure patterns in results.

        Returns:
            DemandPlacementResults object with SLA and placement metrics.
        """
        from ngraph.monte_carlo.functions import demand_placement_analysis
        from ngraph.monte_carlo.results import DemandPlacementResults

        # Convert TrafficMatrixSet to serializable format if needed
        if hasattr(demands_config, "demands") and not isinstance(demands_config, list):
            # This is a TrafficMatrixSet - convert to config list
            serializable_demands = []
            for demand in demands_config.demands:  # type: ignore
                config = {
                    "source_path": demand.source_path,
                    "sink_path": demand.sink_path,
                    "demand": demand.demand,
                    "mode": getattr(demand, "mode", "pairwise"),
                    "flow_policy_config": getattr(demand, "flow_policy_config", None),
                    "priority": getattr(demand, "priority", 0),
                }
                serializable_demands.append(config)
            demands_config = serializable_demands

        raw_results = self.run_monte_carlo_analysis(
            analysis_func=demand_placement_analysis,
            iterations=iterations,
            parallelism=parallelism,
            baseline=baseline,
            seed=seed,
            store_failure_patterns=store_failure_patterns,
            demands_config=demands_config,
            placement_rounds=placement_rounds,
            **kwargs,
        )

        # Process failure patterns if requested
        failure_patterns = {}
        if store_failure_patterns and raw_results["failure_patterns"]:
            failure_patterns = self._build_demand_placement_failure_patterns(
                raw_results["failure_patterns"], raw_results["results"]
            )

        # Extract baseline if present
        baseline_result = None
        if baseline and raw_results["results"]:
            # Baseline is the first result when baseline=True
            baseline_result = raw_results["results"][0]

        return DemandPlacementResults(
            raw_results=raw_results,
            iterations=iterations,
            baseline=baseline_result,
            failure_patterns=failure_patterns,
            metadata=raw_results["metadata"],
        )

    def run_sensitivity_monte_carlo(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        iterations: int = 100,
        parallelism: int = 1,
        shortest_path: bool = False,
        flow_placement: FlowPlacement | str = FlowPlacement.PROPORTIONAL,
        baseline: bool = False,
        seed: int | None = None,
        store_failure_patterns: bool = False,
        **kwargs,
    ) -> Any:  # Will be SensitivityResults when imports are enabled
        """Analyze component criticality for flow capacity under failures.

        Ranks network components by their impact on flow capacity when
        they fail, across Monte Carlo failure scenarios.

        Args:
            source_path: Regex pattern for source node groups.
            sink_path: Regex pattern for sink node groups.
            mode: "combine" (aggregate) or "pairwise" (individual flows).
            iterations: Number of failure scenarios to simulate.
            parallelism: Number of parallel workers (auto-adjusted if needed).
            shortest_path: Whether to use shortest paths only.
            flow_placement: Flow placement strategy.
            baseline: Whether to include baseline (no failures) iteration.
            seed: Optional seed for reproducible results.
            store_failure_patterns: Whether to store failure patterns in results.

        Returns:
            SensitivityResults object with component criticality rankings.
        """
        from ngraph.monte_carlo.functions import sensitivity_analysis
        from ngraph.monte_carlo.results import SensitivityResults

        # Convert string flow_placement to enum if needed
        if isinstance(flow_placement, str):
            try:
                flow_placement = FlowPlacement[flow_placement.upper()]
            except KeyError:
                valid_values = ", ".join([e.name for e in FlowPlacement])
                raise ValueError(
                    f"Invalid flow_placement '{flow_placement}'. "
                    f"Valid values are: {valid_values}"
                ) from None

        raw_results = self.run_monte_carlo_analysis(
            analysis_func=sensitivity_analysis,
            iterations=iterations,
            parallelism=parallelism,
            baseline=baseline,
            seed=seed,
            store_failure_patterns=store_failure_patterns,
            source_regex=source_path,
            sink_regex=sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=flow_placement,
            **kwargs,
        )

        # Process sensitivity results to aggregate component scores
        component_scores = self._process_sensitivity_results(raw_results["results"])

        # Process failure patterns if requested
        failure_patterns = {}
        if store_failure_patterns and raw_results["failure_patterns"]:
            failure_patterns = self._build_sensitivity_failure_patterns(
                raw_results["failure_patterns"], raw_results["results"]
            )

        # Extract baseline if present
        baseline_result = None
        if baseline and raw_results["results"]:
            # Baseline is the first result when baseline=True
            baseline_result = raw_results["results"][0]

        return SensitivityResults(
            raw_results=raw_results,
            iterations=iterations,
            baseline=baseline_result,
            component_scores=component_scores,
            failure_patterns=failure_patterns,
            source_pattern=source_path,
            sink_pattern=sink_path,
            mode=mode,
            metadata=raw_results["metadata"],
        )
