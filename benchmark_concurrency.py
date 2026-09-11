"""
CellOS v0.1 - concurrency / contention benchmark

Tests the actual architectural claim: a shared bus serializes ALL
concurrent traffic through one resource, while a decentralized mesh
only serializes traffic that happens to collide on the same cell at
the same moment.

This is a structural discrete-event simulation (see fabric_batch_completion
and bus_batch_completion in cellos/__init__.py for the exact, stated
modeling assumptions) — not a raw wall-clock parallel-hardware measurement.
"""

import random
import statistics
import json

from cellos import Fabric, CentralBus, random_coord, fabric_batch_completion, bus_batch_completion

GRID = (16, 16)
CONCURRENCY_LEVELS = [1, 4, 16, 64, 128]
TRIALS = 50
SEED = 42


def main():
    rng = random.Random(SEED)
    fabric = Fabric(dim=GRID)
    bus = CentralBus(dim=GRID)

    results = []
    for k in CONCURRENCY_LEVELS:
        fabric_ticks = []
        bus_ticks = []
        contended = []
        for _ in range(TRIALS):
            events = []
            while len(events) < k:
                s = random_coord(GRID, rng)
                t = random_coord(GRID, rng)
                if s != t:
                    events.append((s, t))

            f = fabric_batch_completion(fabric, events)
            b = bus_batch_completion(bus, events)

            fabric_ticks.append(f["max_completion_ticks"])
            bus_ticks.append(b["max_completion_ticks"])
            contended.append(f["contended_cells"])

        results.append({
            "concurrent_events": k,
            "grid": f"{GRID[0]}x{GRID[1]}",
            "fabric_mean_completion_ticks": statistics.mean(fabric_ticks),
            "bus_mean_completion_ticks": statistics.mean(bus_ticks),
            "fabric_mean_contended_cells": statistics.mean(contended),
            "speedup_x": statistics.mean(bus_ticks) / statistics.mean(fabric_ticks),
        })

    print(f"{'Concurrent':<12}{'Fabric ticks':<15}{'Bus ticks':<12}{'Contended cells':<18}{'Bus/Fabric ratio'}")
    for r in results:
        print(
            f"{r['concurrent_events']:<12}"
            f"{r['fabric_mean_completion_ticks']:<15.2f}"
            f"{r['bus_mean_completion_ticks']:<12.2f}"
            f"{r['fabric_mean_contended_cells']:<18.2f}"
            f"{r['speedup_x']:.2f}x"
        )

    with open("benchmark_concurrency_results.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
