"""Configuration classes for NetGraph components."""

from dataclasses import dataclass


@dataclass
class TrafficManagerConfig:
    """Configuration for traffic demand placement estimation."""

    # Default number of placement rounds when no data is available
    default_rounds: int = 5

    # Minimum number of placement rounds
    min_rounds: int = 5

    # Maximum number of placement rounds
    max_rounds: int = 100

    # Multiplier for ratio-based round estimation
    ratio_base: int = 5
    ratio_multiplier: int = 5

    def estimate_rounds(self, demand_capacity_ratio: float) -> int:
        """Calculate placement rounds based on demand to capacity ratio."""
        estimated = int(self.ratio_base + self.ratio_multiplier * demand_capacity_ratio)
        return max(self.min_rounds, min(estimated, self.max_rounds))


# Global configuration instance
TRAFFIC_CONFIG = TrafficManagerConfig()
