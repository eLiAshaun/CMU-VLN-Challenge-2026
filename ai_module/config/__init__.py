"""Central configuration and state management package.

This package provides the Single Source of Truth for all configuration
and explicit runtime state management.

Usage:

    from config.loader import load_config
    from config.runtime_state import create_session_context

    # At application startup
    config = load_config(
        asset_manifest_path=Path("configs/model_assets.json")
    )

    # Configuration is now immutable
    print(config.qwen3vl.quantization)  # "int8"

    # Create session context for runtime state
    session = create_session_context()

    # Register workers
    session.workers.register("qwen3vl", "@mast3r_qwen3vl", pid=12345)
    session.workers.mark_ready("qwen3vl")

    # Create episode when question arrives
    episode = session.create_episode(
        question="How many chairs are there?",
        run_dir=Path("/runs/episode_123"),
        deadline=time.monotonic() + 600.0,
    )

    # Explicit state transitions
    episode.transition_to(EpisodeState.PREFLIGHT)
    episode.transition_to(EpisodeState.READY)
    episode.transition_to(EpisodeState.RUNNING)

    # When episode completes
    session.cleanup_episode(mark_as="completed")

Module structure:

- schema.py: Typed configuration schema with validation
- loader.py: Configuration loading with precedence rules
- runtime_state.py: Episode and worker state management
"""

from .loader import (
    load_config,
    load_config_for_worker,
    write_effective_config,
)
from .runtime_state import (
    EpisodeContext,
    SessionContext,
    WorkerInfo,
    WorkerRegistry,
    create_session_context,
)
from .schema import (
    AppConfig,
    EpisodeState,
    WorkerStatus,
    create_default_config,
)

__all__ = [
    # Configuration loading
    "load_config",
    "load_config_for_worker",
    "write_effective_config",
    "create_default_config",

    # Configuration schema
    "AppConfig",

    # Runtime state
    "SessionContext",
    "EpisodeContext",
    "WorkerRegistry",
    "WorkerInfo",
    "create_session_context",

    # Enums
    "EpisodeState",
    "WorkerStatus",
]
