"""Fail-closed semantic execution over persistent MASt3R observations."""

from .query_executor import execute_task
from .root_finalizer import finalize_execution
from .world_model import update_world_model

__all__ = ["execute_task", "finalize_execution", "update_world_model"]
