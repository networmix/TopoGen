"""Deterministic seed derivation to avoid global random.seed() order dependencies."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional


class SeedManager:
    """Manages deterministic seed derivation for isolated component reproducibility.

    Global random.seed() creates order dependencies and component interference.
    SeedManager derives unique seeds per component from a master seed using SHA-256,
    ensuring reproducible results regardless of execution order or parallelism.

    Usage:
        seed_mgr = SeedManager(42)
        failure_seed = seed_mgr.derive_seed("failure_policy", "default")
        analysis_seed = seed_mgr.derive_seed("workflow", "capacity_analysis", 0)
    """

    def __init__(self, master_seed: Optional[int] = None) -> None:
        """Initialize the seed manager.

        Args:
            master_seed: Master seed for deterministic operations. If None,
                        seed derivation will return None (non-deterministic).
        """
        self.master_seed = master_seed

    def derive_seed(self, *components: Any) -> Optional[int]:
        """Derive a deterministic seed from master seed and component identifiers.

        Uses a hash-based approach to generate consistent seeds for different
        components while ensuring good distribution of seed values.

        Args:
            *components: Component identifiers (strings, integers, etc.) that
                        uniquely identify the component needing a seed.

        Returns:
            Derived seed as positive integer, or None if no master seed set.

        Example:
            seed_mgr = SeedManager(42)
            policy_seed = seed_mgr.derive_seed("failure_policy", "default")
            worker_seed = seed_mgr.derive_seed("capacity_analysis", "worker", 3)
        """
        if self.master_seed is None:
            return None

        # Create a deterministic hash from master seed and components
        seed_input = f"{self.master_seed}:" + ":".join(str(c) for c in components)
        hash_digest = hashlib.sha256(seed_input.encode()).digest()

        # Convert first 4 bytes to a positive integer
        seed_value = int.from_bytes(hash_digest[:4], byteorder="big")
        return seed_value & 0x7FFFFFFF  # Ensure positive 32-bit integer

    def create_random_state(self, *components: Any) -> random.Random:
        """Create a new Random instance with derived seed.

        Args:
            *components: Component identifiers for seed derivation.

        Returns:
            New Random instance seeded with derived seed, or unseeded if no master seed.
        """
        derived_seed = self.derive_seed(*components)
        rng = random.Random()
        if derived_seed is not None:
            rng.seed(derived_seed)
        return rng

    def seed_global_random(self, *components: Any) -> None:
        """Seed the global random module with derived seed.

        Args:
            *components: Component identifiers for seed derivation.
        """
        derived_seed = self.derive_seed(*components)
        if derived_seed is not None:
            random.seed(derived_seed)
