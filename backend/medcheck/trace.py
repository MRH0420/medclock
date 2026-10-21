"""运行轨迹。评委问"你的 Agent 到底做了什么"时，把 traces/*.jsonl 打开给他们看。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import TRACE_DIR


def _fallback(obj: Any) -> Any:
    """轨迹落盘绝不能因为一个不可序列化对象而炸掉主流程。

    Note / Evidence 这类小对象统一降级成它们的文字形式。
    """
    for attr in ("text", "zh"):                     # Note.text(lang) / Note.zh
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    return repr(obj)


class Tracer:
    """只追加的事件流：observe / tool_call / screen / safety_gate / escalate / write / finish。"""

    def __init__(self, enabled: bool = True, run_id: str | None = None) -> None:
        self.enabled = enabled
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.step = 0
        self.events: list[dict[str, Any]] = []
        self.path: Path | None = None

    def log(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.step += 1
        event = {
            "run_id": self.run_id,
            "step": self.step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "payload": payload or {},
        }
        self.events.append(event)
        if self.enabled:
            self._flush(event)
        return event

    def _flush(self, event: dict[str, Any]) -> None:
        try:
            path = self.path or (TRACE_DIR / f"{self.run_id}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=_fallback) + "\n")
            self.path = path
        except (OSError, TypeError, ValueError):
            self.enabled = False

    def summary(self) -> list[str]:
        out = []
        for e in self.events:
            out.append(f"[{e['step']:02d}] {e['kind']}: {json.dumps(e['payload'], ensure_ascii=False)[:160]}")
        return out
