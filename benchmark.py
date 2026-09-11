"""
CellOS v0.1 - benchmark harness

Runs repeated trials of Fabric (decentralized mesh) vs CentralBus
(shared-bus baseline) over the same random source/target pairs, on the
same grid, in the same process. Prints real measured results.

This is a SOFTWARE SIMULATION benchmark. It measures the relative
behavior of two routing topologies implemented in Python on whatever
machine runs this script — not a hardware or silicon benchmark.
"""

import random
import statistics
import json
import sys

from cellos import Fabric, CentralBus, random_coord

TRIALS_PER_CONFIG = 200
GRID_SIZES = [(4, 4), (8, 8), (16, 16)]
SEED = 42


def run_config(dim):
    rng = random.Random(SEED)
    fabric = Fabric(dim=dim)
    bus = CentralBus(dim=dim)

    fabric_latencies = []
    bus_latencies = []
    fabric_hops = []
    fabric_active_fractions = []
    bus_active_fractions = []

    for _ in range(TRIALS_PER_CONFIG):
        source = random_coord(dim, rng)
        target = random_coord(dim, rng)
        if source == target:
            continue

        f = fabric.pulse(source=source, target=target)
        b = bus.pulse(source=source, target=target)

        fabric_latencies.append(f["elapsed_seconds"])
        bus_latencies.append(b["elapsed_seconds"])
        fabric_hops.append(f["hops"])
        fabric_active_fractions.append(f["active_fraction"])
        bus_active_fractions.append(b["active_fraction"])

    return {
        "grid": f"{dim[0]}x{dim[1]}",
        "trials": len(fabric_latencies),
        "fabric_latency_us_mean": statistics.mean(fabric_latencies) * 1e6,
        "fabric_latency_us_median": statistics.median(fabric_latencies) * 1e6,
        "bus_latency_us_mean": statistics.mean(bus_latencies) * 1e6,
        "bus_latency_us_median": statistics.median(bus_latencies) * 1e6,
        "fabric_mean_hops": statistics.mean(fabric_hops),
        "fabric_mean_active_fraction": statistics.mean(fabric_active_fractions),
        "bus_mean_active_fraction": statistics.mean(bus_active_fractions),
    }


def main():
    results = [run_config(dim) for dim in GRID_SIZES]

    print(f"{'Grid':<8}{'Trials':<8}{'Fabric us (mean/median)':<26}{'Bus us (mean/median)':<24}{'Mean hops':<12}{'Fabric active %':<18}{'Bus active %'}")
    for r in results:
        print(
            f"{r['grid']:<8}"
            f"{r['trials']:<8}"
            f"{r['fabric_latency_us_mean']:.2f}/{r['fabric_latency_us_median']:.2f}".ljust(26)
            + f"{r['bus_latency_us_mean']:.2f}/{r['bus_latency_us_median']:.2f}".ljust(24)
            + f"{r['fabric_mean_hops']:.2f}".ljust(12)
            + f"{r['fabric_mean_active_fraction']*100:.1f}%".ljust(18)
            + f"{r['bus_mean_active_fraction']*100:.1f}%"
        )

    with open("benchmark_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nRaw results written to benchmark_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
