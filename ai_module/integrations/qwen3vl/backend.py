"""Lazy Qwen3-VL semantic grounding and verification adapter."""

class Qwen3VLBackend:
    def __init__(self, implementation=None):
        self.implementation = implementation

    def execute(self, request):
        if request.operation == "healthcheck":
            if self.implementation is None:
                raise RuntimeError("qwen3vl_checkpoint_unavailable")
            return self.implementation.healthcheck()
        if request.operation not in {
            "ground_objects",
            "ground_objects_batch",
            "verify_object",
            "verify_object_batch",
            "verify_relation",
            "verify_relation_batch",
            "rank_candidates",
        }:
            raise ValueError("qwen3vl_is_verification_only")
        if self.implementation is None:
            raise RuntimeError("qwen3vl_checkpoint_unavailable")
        return self.implementation(request)
