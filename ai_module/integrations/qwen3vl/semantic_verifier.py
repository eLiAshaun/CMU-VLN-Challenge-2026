"""Calibrated, evidence-only semantic verification for ambiguous objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SemanticVerification:
    target_probability: float
    anchor_probability: float
    confuser_probabilities: Mapping[str, float] = field(default_factory=dict)
    rationale_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = {
            "target_probability": self.target_probability,
            "anchor_probability": self.anchor_probability,
            **dict(self.confuser_probabilities),
        }
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in values.values()):
            raise ValueError("semantic probabilities must be finite and in [0, 1]")

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "SemanticVerification":
        base = {
            "target_probability",
            "anchor_probability",
            "confuser_probabilities",
            "rationale_tags",
        }
        if set(payload) != frozenset(base):
            raise ValueError("semantic verifier output must follow the strict JSON schema")
        return cls(
            target_probability=float(payload["target_probability"]),
            anchor_probability=float(payload["anchor_probability"]),
            confuser_probabilities={str(key): float(value) for key, value in dict(payload["confuser_probabilities"]).items()},
            rationale_tags=tuple(map(str, payload["rationale_tags"])),
        )


class SemanticVerifier:
    """Fuse proposal and discriminative evidence; VLM is ambiguous-only."""

    def verify(
        self,
        *,
        proposal_probabilities: Sequence[float],
        discriminative_probability: float | None,
        vlm_verification: SemanticVerification | None = None,
        ambiguity_band: tuple[float, float] = (0.25, 0.75),
    ) -> float:
        proposals = [min(1.0, max(0.0, float(value))) for value in proposal_probabilities]
        if not proposals and discriminative_probability is None:
            return 0.5
        proposal = 1.0
        for value in proposals:
            proposal *= 1.0 - value
        proposal = 1.0 - proposal
        if discriminative_probability is None:
            posterior = proposal
        else:
            discriminative = min(1.0, max(0.0, float(discriminative_probability)))
            posterior = 0.45 * proposal + 0.55 * discriminative
        low, high = ambiguity_band
        if vlm_verification is not None and low <= posterior <= high:
            posterior = 0.70 * posterior + 0.30 * vlm_verification.target_probability
        return min(1.0, max(0.0, posterior))
