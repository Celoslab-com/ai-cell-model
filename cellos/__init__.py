"""
CellOS Virtual Mesh Engine (v0.1)
----------------------------------
A minimal, honest software emulator for the AI Cell model:

    AI Cell = Compute + Memory + State + Routing

This module implements two comparable execution models over the SAME
grid layout, so that any benchmark comparing them is a fair, same-machine
software simulation:

  - Fabric:      a decentralized mesh of Cells. Each Cell only knows its
                 four grid neighbors. An event travels hop-by-hop across
                 the mesh (peer-to-peer), the way the routing diagram on
                 the CellOS site describes.

  - CentralBus:  a baseline model of a traditional shared-bus / monolithic
                 architecture, where every event must pass through one
                 central dispatcher regardless of where source and target
                 sit on the grid.

IMPORTANT — what this is and is not:
This measures REAL wall-clock behavior of REAL Python code running on
whatever machine executes it. It is a software simulation of a routing
topology, not a hardware benchmark. Absolute microsecond numbers will
vary by machine and mean nothing in isolation — the meaningful signal is
the RELATIVE comparison between Fabric and CentralBus under identical
conditions, and how that comparison changes with grid size and locality.
Any hardware/silicon latency or power claims are future work, not
measured here.
"""

from __future__ import annotations

import time
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cell:
    """A single AI Cell: local compute, local memory, local state."""

    x: int
    y: int
    memory: dict = field(default_factory=dict)
    state: str = "idle"
    neighbors: list = field(default_factory=list)

    @property
    def coord(self) -> tuple[int, int]:
        return (self.x, self.y)

    def receive(self, payload) -> None:
        """Local compute step: a Cell processing an event touches only
        its own memory/state — never a shared/global resource."""
        self.state = "active"
        self.memory["last_payload"] = payload


class Fabric:
    """A decentralized N x M mesh of Cells connected to their 4 grid
    neighbors only. Routing between any two cells is peer-to-peer,
    hop-by-hop, via breadth-first shortest path across the mesh graph —
    no cell has global knowledge of the whole fabric.
    """

    def __init__(self, dim: tuple[int, int] = (8, 8)):
        self.width, self.height = dim
        self.cells: dict[tuple[int, int], Cell] = {
            (x, y): Cell(x, y)
            for x in range(self.width)
            for y in range(self.height)
        }
        self._wire_neighbors()

    def _wire_neighbors(self) -> None:
        for (x, y), cell in self.cells.items():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + dx, y + dy)
                if n in self.cells:
                    cell.neighbors.append(n)

    def _reset_states(self) -> None:
        for cell in self.cells.values():
            cell.state = "idle"

    def shortest_path(self, source: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
        """Breadth-first search across the neighbor graph — this is the
        only routing information a real decentralized mesh would have:
        no shortcuts, no global coordinate lookup, just hop-by-hop."""
        if source == target:
            return [source]
        visited = {source}
        queue = deque([[source]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            for n in self.cells[node].neighbors:
                if n in visited:
                    continue
                new_path = path + [n]
                if n == target:
                    return new_path
                visited.add(n)
                queue.append(new_path)
        raise ValueError(f"No path between {source} and {target}")

    def pulse(self, source: Optional[tuple[int, int]] = None,
              target: Optional[tuple[int, int]] = None,
              payload=None) -> dict:
        """Route one event from source to target, hop-by-hop through the
        mesh. If no source/target given, pulse the fabric from its
        center outward as a demo wave (matches `fabric.pulse()` with no
        args in the quickstart snippet).

        Returns real measured metrics for this single event.
        """
        self._reset_states()

        if source is None or target is None:
            source = (self.width // 2, self.height // 2)
            target = (0, 0)

        start = time.perf_counter()
        path = self.shortest_path(source, target)
        for coord in path:
            self.cells[coord].receive(payload)
        elapsed = time.perf_counter() - start

        active = sum(1 for c in self.cells.values() if c.state == "active")
        total = len(self.cells)

        return {
            "model": "fabric",
            "source": source,
            "target": target,
            "hops": len(path) - 1,
            "path": path,
            "elapsed_seconds": elapsed,
            "cells_activated": active,
            "cells_total": total,
            "active_fraction": active / total,
        }


class CentralBus:
    """Baseline comparison model: same grid of cells and coordinates as
    Fabric, but every event is routed through a single central hub node
    regardless of source/target position — the way a shared bus or a
    monolithic controller would. Used only to produce a fair, same-code,
    same-machine comparison against Fabric.
    """

    def __init__(self, dim: tuple[int, int] = (8, 8)):
        self.width, self.height = dim
        self.cells: dict[tuple[int, int], Cell] = {
            (x, y): Cell(x, y)
            for x in range(self.width)
            for y in range(self.height)
        }
        self.hub = (self.width // 2, self.height // 2)

    def _reset_states(self) -> None:
        for cell in self.cells.values():
            cell.state = "idle"

    def pulse(self, source: tuple[int, int], target: tuple[int, int], payload=None) -> dict:
        self._reset_states()
        start = time.perf_counter()

        # every event passes through the hub, then to the target —
        # and, modeling a shared bus, EVERY cell's shared resource
        # (the bus itself) is considered "touched"/active for the
        # duration of the transfer, not just the cells on the path.
        path = [source, self.hub, target]
        for coord in path:
            self.cells[coord].receive(payload)
        elapsed = time.perf_counter() - start

        for c in self.cells.values():
            c.state = "active"  # shared bus: whole fabric's bus access is occupied

        active = len(self.cells)
        total = len(self.cells)

        return {
            "model": "central_bus",
            "source": source,
            "target": target,
            "hops": len(path) - 1,
            "path": path,
            "elapsed_seconds": elapsed,
            "cells_activated": active,
            "cells_total": total,
            "active_fraction": active / total,
        }


def random_coord(dim: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    return (rng.randrange(dim[0]), rng.randrange(dim[1]))


def fabric_batch_completion(fabric: "Fabric", events: list[tuple[tuple, tuple]]) -> dict:
    """Structural model of concurrent event completion for a decentralized
    mesh: each event's path is computed independently. Cells are assumed
    to execute independently in parallel UNLESS two events need the same
    cell at the same hop index (real contention), in which case those
    events queue at that cell.

    This is a discrete-event STRUCTURAL simulation, not a raw wall-clock
    multi-threaded measurement (Python's GIL makes that meaningless for
    this kind of workload). The assumption — independent cells proceed
    concurrently, shared cells serialize — is the actual architectural
    claim being tested, stated explicitly rather than implied.
    """
    paths = [fabric.shortest_path(s, t) for s, t in events]
    max_len = max(len(p) for p in paths)

    # occupancy[hop_index][cell] = number of events wanting that cell
    # at that hop; if >1, those events serialize (queue) at that hop.
    completion_time = [0] * len(paths)
    cell_free_at = {}  # cell -> hop index at which it becomes free again

    for hop in range(max_len):
        for i, path in enumerate(paths):
            if hop >= len(path):
                continue
            cell = path[hop]
            earliest = max(hop, cell_free_at.get(cell, 0))
            completion_time[i] = earliest + 1
            cell_free_at[cell] = earliest + 1

    contended_cells = sum(1 for c, t in cell_free_at.items() if t > 1)

    return {
        "model": "fabric_batch",
        "n_events": len(events),
        "max_completion_ticks": max(completion_time),
        "mean_completion_ticks": statistics.mean(completion_time),
        "contended_cells": contended_cells,
    }


def bus_batch_completion(bus: "CentralBus", events: list[tuple[tuple, tuple]]) -> dict:
    """Structural model for a shared bus: every event must pass through
    the single hub, one at a time (the hub is a single shared resource,
    so events queue/serialize there regardless of where they start or
    end). Each event takes 2 hops (source->hub, hub->target); the hub
    can only service one hop-step per tick.
    """
    n = len(events)
    ticks_needed = 2 * n  # hub services exactly one hop-step per tick
    return {
        "model": "bus_batch",
        "n_events": n,
        "max_completion_ticks": ticks_needed,
        "mean_completion_ticks": ticks_needed,  # all wait on the same shared resource
        "contended_cells": 1,  # the hub itself
    }
