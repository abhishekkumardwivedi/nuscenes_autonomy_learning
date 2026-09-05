from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class PipelineContext:
    """A transparent container passed from one stage to the next.

    Instead of hiding state inside a large class, every stage writes named values
    here. This makes it easy to inspect exactly what each stage produced.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        self.data.update(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def require(self, *keys: str) -> None:
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise KeyError(
                "Pipeline stage dependency missing: " + ", ".join(missing)
            )

    def keys(self) -> Iterable[str]:
        return self.data.keys()
