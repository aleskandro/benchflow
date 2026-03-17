from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import ResolvedRunPlan


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    name: str
    namespace: str
    experiment: str
    platform: str
    mode: str
    backend: str
    status: str
    finished: bool
    succeeded: bool
    start_time: str
    completion_time: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "experiment": self.experiment,
            "platform": self.platform,
            "mode": self.mode,
            "backend": self.backend,
            "status": self.status,
            "finished": self.finished,
            "succeeded": self.succeeded,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "message": self.message,
        }


class ExecutionBackend(Protocol):
    name: str

    def render_run(
        self,
        plan: ResolvedRunPlan,
        *,
        execution_name: str,
        setup_mode: str,
        teardown: bool,
    ) -> dict[str, Any]: ...

    def render_matrix(
        self,
        plans: list[ResolvedRunPlan],
        *,
        execution_name: str,
        child_execution_name: str,
    ) -> dict[str, Any]: ...

    def get(self, namespace: str, name: str) -> dict[str, Any] | None: ...

    def list(
        self, namespace: str, *, label_selector: str = ""
    ) -> list[dict[str, Any]]: ...

    def summarize(self, resource: dict[str, Any]) -> ExecutionSummary: ...

    def cancel(self, namespace: str, name: str) -> None: ...

    def follow(self, namespace: str, name: str, *, poll_interval: int = 5) -> bool: ...
