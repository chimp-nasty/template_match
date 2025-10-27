# utils/prof.py
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Tuple, Dict, Iterable

@dataclass
class Profiler:
    enabled: bool = True
    events: List[Tuple[str, float]] = field(default_factory=list)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            t1 = perf_counter()
            self.events.append((name, (t1 - t0) * 1000.0))  # ms

    def record(self, name: str, ms: float):
        if self.enabled:
            self.events.append((name, ms))

    def clear(self):
        self.events.clear()

    def totals(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in self.events:
            out[k] = out.get(k, 0.0) + v
        return out

    def summary_lines(self, group_prefixes: Iterable[str] = ()):
        # If you pass group prefixes (e.g. ["tm.t"]), we’ll show grouped subtotal too
        totals = self.totals()
        lines = []
        # Detailed lines in the order they were recorded
        for k, v in self.events:
            lines.append(f"{k:34s} : {v:8.3f} ms")
        # Optional subtotal groups
        for p in group_prefixes:
            s = sum(v for k, v in totals.items() if k.startswith(p))
            lines.append(f"{p+'* subtotal':34s} : {s:8.3f} ms")
        # Overall total
        overall = sum(v for _, v in self.events)
        lines.append(f"{'TOTAL':34s} : {overall:8.3f} ms")
        return lines

    def summary_str(self, group_prefixes: Iterable[str] = ()):
        return "\n".join(self.summary_lines(group_prefixes))
