"""Runtime state management - Episode and worker lifecycle.

This module defines the state management layer separate from configuration.
State is mutable and scoped to specific entities (episode, worker, session).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import EpisodeState, WorkerStatus


@dataclass
class WorkerInfo:
    """Runtime state for a single worker process."""
    name: str
    endpoint: str
    pid: int | None
    status: WorkerStatus
    started_at: float = field(default_factory=time.monotonic)
    last_request_at: float | None = None
    request_count: int = 0
    error_count: int = 0
    last_error: str | None = None


@dataclass
class WorkerRegistry:
    """Central registry of all worker processes.

    Tracks worker health and lifecycle.
    Provides query interface for "is worker X healthy?".
    """
    workers: dict[str, WorkerInfo] = field(default_factory=dict)

    def register(
        self,
        name: str,
        endpoint: str,
        pid: int | None = None,
    ) -> None:
        """Register a new worker."""
        self.workers[name] = WorkerInfo(
            name=name,
            endpoint=endpoint,
            pid=pid,
            status=WorkerStatus.STARTING,
        )

    def mark_ready(self, name: str) -> None:
        """Mark worker as ready to accept requests."""
        if name in self.workers:
            self.workers[name].status = WorkerStatus.READY

    def mark_busy(self, name: str) -> None:
        """Mark worker as busy processing request."""
        if name in self.workers:
            self.workers[name].status = WorkerStatus.BUSY
            self.workers[name].last_request_at = time.monotonic()
            self.workers[name].request_count += 1

    def mark_failed(self, name: str, error: str) -> None:
        """Mark worker as failed."""
        if name in self.workers:
            self.workers[name].status = WorkerStatus.FAILED
            self.workers[name].error_count += 1
            self.workers[name].last_error = error

    def is_healthy(self, name: str) -> bool:
        """Check if worker is healthy and ready."""
        worker = self.workers.get(name)
        return worker is not None and worker.status == WorkerStatus.READY

    def is_enabled(self, name: str) -> bool:
        """Check if worker exists in registry (was started)."""
        return name in self.workers

    def get_status(self, name: str) -> WorkerStatus | None:
        """Get current worker status."""
        worker = self.workers.get(name)
        return worker.status if worker else None


@dataclass
class EpisodeContext:
    """Runtime state for a single episode.

    Each episode is a self-contained execution unit with:
    - Unique ID
    - Explicit lifecycle state
    - Deadline tracking
    - Result accumulation

    Episode state is reset completely between episodes.
    No episode state should leak into the next episode.
    """
    episode_id: str
    question: str
    state: EpisodeState
    started_at: float
    deadline: float

    # Paths
    run_dir: Path
    scene_memory_path: Path | None = None

    # Accumulated data
    reconstruction_keyframes: list[dict[str, Any]] = field(default_factory=list)
    chain_durations: list[float] = field(default_factory=list)
    navigation_durations: list[float] = field(default_factory=list)
    navigation_started: dict[str, float] = field(default_factory=dict)
    navigation_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Status flags
    episode_terminalized: bool = False

    # State transition history (for debugging)
    state_history: list[tuple[float, EpisodeState]] = field(default_factory=list)

    def is_timeout(self) -> bool:
        """Check if episode has exceeded deadline."""
        return time.monotonic() > self.deadline

    def remaining_time(self) -> float:
        """Get remaining time before deadline (can be negative)."""
        return self.deadline - time.monotonic()

    def transition_to(self, new_state: EpisodeState) -> None:
        """Transition to new state with validation.

        Raises:
            ValueError: If transition is invalid
        """
        # Define valid transitions
        valid_transitions = {
            (EpisodeState.IDLE, EpisodeState.QUESTION_RECEIVED),
            (EpisodeState.QUESTION_RECEIVED, EpisodeState.PREFLIGHT),
            (EpisodeState.PREFLIGHT, EpisodeState.READY),
            (EpisodeState.READY, EpisodeState.RUNNING),
            (EpisodeState.RUNNING, EpisodeState.FINISHING),
            (EpisodeState.RUNNING, EpisodeState.TIMEOUT),
            (EpisodeState.RUNNING, EpisodeState.FAILED),
            (EpisodeState.FINISHING, EpisodeState.IDLE),
            (EpisodeState.TIMEOUT, EpisodeState.IDLE),
            (EpisodeState.FAILED, EpisodeState.IDLE),
            (EpisodeState.QUESTION_RECEIVED, EpisodeState.CANCELLED),
            (EpisodeState.PREFLIGHT, EpisodeState.CANCELLED),
            (EpisodeState.READY, EpisodeState.CANCELLED),
            (EpisodeState.RUNNING, EpisodeState.CANCELLED),
            (EpisodeState.CANCELLED, EpisodeState.IDLE),
        }

        if (self.state, new_state) not in valid_transitions:
            raise ValueError(
                f"Invalid state transition: {self.state.value} -> {new_state.value}"
            )

        # Record transition
        self.state_history.append((time.monotonic(), self.state))
        self.state = new_state

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode state for persistence."""
        return {
            "episode_id": self.episode_id,
            "question": self.question,
            "state": self.state.value,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "run_dir": str(self.run_dir),
            "reconstruction_keyframe_count": len(self.reconstruction_keyframes),
            "chain_duration_count": len(self.chain_durations),
            "navigation_attempt_count": len(self.navigation_attempts),
            "episode_terminalized": self.episode_terminalized,
            "state_history": [
                {"time": t, "state": s.value}
                for t, s in self.state_history
            ],
        }


@dataclass
class SessionContext:
    """Session-level runtime state.

    Lives for the entire container/process lifetime.
    Contains long-lived resources and cross-episode statistics.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)

    # Current episode (None when idle)
    current_episode: EpisodeContext | None = None

    # Worker registry
    workers: WorkerRegistry = field(default_factory=WorkerRegistry)

    # Session statistics
    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    timeout_episodes: int = 0

    def is_idle(self) -> bool:
        """Check if session is idle (no active episode)."""
        return self.current_episode is None

    def is_busy(self) -> bool:
        """Check if session is busy (episode active)."""
        return self.current_episode is not None

    def create_episode(
        self,
        question: str,
        run_dir: Path,
        deadline: float,
    ) -> EpisodeContext:
        """Create and activate a new episode.

        Raises:
            RuntimeError: If episode already active
        """
        if self.current_episode is not None:
            raise RuntimeError(
                f"Cannot create episode: episode {self.current_episode.episode_id} "
                f"already active"
            )

        episode_id = f"{uuid.uuid4().hex}"

        episode = EpisodeContext(
            episode_id=episode_id,
            question=question,
            state=EpisodeState.QUESTION_RECEIVED,
            started_at=time.monotonic(),
            deadline=deadline,
            run_dir=run_dir,
        )

        self.current_episode = episode
        self.total_episodes += 1

        return episode

    def cleanup_episode(self, mark_as: str = "completed") -> None:
        """Clean up current episode and reset to idle.

        Args:
            mark_as: How to mark episode ("completed", "failed", "timeout")
        """
        if self.current_episode is None:
            return

        # Update statistics
        if mark_as == "completed":
            self.successful_episodes += 1
        elif mark_as == "failed":
            self.failed_episodes += 1
        elif mark_as == "timeout":
            self.timeout_episodes += 1

        # Reset current episode
        self.current_episode = None


def create_session_context() -> SessionContext:
    """Create a new session context."""
    return SessionContext()
