"""Update strategies for ensemble parameter updates.

This package provides different update strategies for parameter updates,
allowing different update methods (standard ES, adaptive localization,
distance-based localization) to be applied to different parameters.

Strategy Lifecycle:
    1. Create strategy instances with dependencies (rng, settings, callback)
    2. Call strategy.prepare(obs_context) to initialize with observation data
    3. Call strategy.update() for each parameter group

Example usage:
    from ert.analysis import build_update_strategy_map
    from ert.analysis._update_strategies import (
        StandardESUpdate,
        AdaptiveLocalizationUpdate,
        ObservationContext,
    )

    # Option 1: Use the factory to build strategies from ESSettings
    strategy_map = build_update_strategy_map(
        es_settings, parameters, param_configs, rng, progress_callback
    )

    # Option 2: Build a custom strategy map for per-parameter control
    standard_strategy = StandardESUpdate(
        settings.inversion, settings.enkf_truncation, rng, progress_callback
    )
    adaptive_strategy = AdaptiveLocalizationUpdate(
        settings.correlation_threshold, rng, progress_callback
    )
    strategy_map = {
        "PORO": adaptive_strategy,
        "PERM": standard_strategy,
    }

    # Pass strategy_map to smoother_update
    smoother_update(..., strategy_map=strategy_map)
"""

from ._adaptive import AdaptiveLocalizationUpdate
from ._distance import DistanceLocalizationUpdate
from ._protocol import (
    ObservationContext,
    ObservationLocations,
    TimedIterator,
    UpdateStrategy,
)
from ._standard import StandardESUpdate

__all__ = [
    "AdaptiveLocalizationUpdate",
    "DistanceLocalizationUpdate",
    "ObservationContext",
    "ObservationLocations",
    "StandardESUpdate",
    "TimedIterator",
    "UpdateStrategy",
]
