# The AI Cell Model

**Decentralized compute and modular fault containment as an alternative to monolithic shared-parameter architectures.**

Working paper by Daysun Simeon — Celos Labs. Software-only evidence, not yet peer reviewed. No patents filed.

## The idea

Most AI systems share a resource — a bus, a parameter set — across many tasks. That's efficient when nothing goes wrong, and fragile when something does: a saturated bus or a corrupted shared model degrades everything it serves at once.

The **AI Cell** is a unit that co-locates compute, memory, state, and routing locally, trading some of that shared-resource efficiency for graceful degradation under load and under failure.

## Repository layout

Intended reading order for a technical reviewer — implementation → benchmark code → raw results → methodology:

| Path | What it is |
|---|---|
| [`celos/__init__.py`](./celos/__init__.py) | Core reference implementation: `Cell`, `Fabric` (decentralized mesh), `CentralBus` (shared-bus baseline), plus the `fabric_batch_completion` / `bus_batch_completion` contention model used by the concurrency benchmark |
| [`benchmark.py`](./benchmark.py) | Single-event, per-grid-size latency benchmark (mesh vs. bus) |
| [`benchmark_results.json`](./benchmark_results.json) | Raw output of `benchmark.py` |
| [`benchmark_concurrency.py`](./benchmark_concurrency.py) | Concurrent-load / contention benchmark (mesh vs. bus under simultaneous events) |
| [`benchmark_concurrency_results.json`](./benchmark_concurrency_results.json) | Raw output of `benchmark_concurrency.py` |
| [`marl_experiment.py`](./marl_experiment.py) | Fault-containment experiment: independent vs. shared Q-tables under injected corruption |
| [`marl_experiment_results.json`](./marl_experiment_results.json) | Raw output of `marl_experiment.py` |
| [`cellos-emulator-v0.1-1.zip`](./cellos-emulator-v0.1-1.zip) | Complete original v0.1 package — the code above, bundled |
| [`celos-ai-cell-whitepaper.md`](./celos-ai-cell-whitepaper.md) / [`.PDF`](./celos-ai-cell-whitepaper.PDF) | Full paper: problem framing, related work, exact hyperparameters/seeds, honest caveats |

**All three experiments are directly browsable and runnable from the repo root — no download or extraction required.** The ZIP remains available as a complete, self-contained snapshot of the v0.1 package.

## What these benchmarks are — and aren't

Every result below comes from running a **Python software model** of a decentralized mesh (`Fabric`) against a **Python software model** of a shared bus (`CentralBus`), in the same process, on whatever machine runs the script — plus one tabular Q-learning experiment. There is no custom hardware, FPGA, or silicon involved anywhere in this repo.

- **Architectural behavior demonstrated:** a shared resource serializes all traffic (or all fault exposure) through one point; a decentralized/independent design only serializes or fails where it actually collides. This is the structural claim the code is built to expose.
- **Actual measurement taken:** Python wall-clock timing (`benchmark.py`), a discrete-event tick count (`benchmark_concurrency.py`), and tabular Q-learning success rates before/after injected corruption (`marl_experiment.py`) — i.e., how these specific Python programs behave, not how any physical accelerator, bus, or chip would perform.
- **Not yet tested:** any hardware timing, power draw, or latency claim, and any failure mode beyond one fixed random-zeroing corruption on one toy task. FPGA emulation — the first hardware-adjacent validation — is planned, not done (see Status).

### `benchmark.py` — single-event latency by grid size

Routes one isolated event through the mesh and through the bus, across increasing grid sizes (200 trials/size, trials where source==target skipped), and records wall-clock latency and mesh hop count. This isolates the "average case": for a single event, the bus's fixed hub-and-spoke dispatch beats the mesh's variable-length path at every size tested — the mesh's own code recomputes shortest path on every call rather than caching it.

| Grid | Fabric mean latency | Bus mean latency | Fabric mean hops | Fabric active cells |
|---|---|---|---|---|
| 4×4 | 4.7 µs | 0.6 µs | 2.6 | 22.3% |
| 8×8 | 14.6 µs | 0.8 µs | 4.9 | 9.3% |
| 16×16 | 61.6 µs | 1.1 µs | 9.9 | 4.2% |

The bus always shows 100% active cells (it models the whole shared resource as occupied for every transfer). The mesh's shrinking active-cell fraction as grid size grows is a real, measured property of the two topologies — a proxy for "only active cells draw power," not a wattage measurement, since no physical hardware exists yet to measure that directly.

### `benchmark_concurrency.py` — concurrent-load contention

A structural discrete-event simulation (`fabric_batch_completion` / `bus_batch_completion` in `celos/__init__.py`) — not a multi-threaded or multi-process hardware benchmark, since Python's GIL makes that comparison meaningless for this workload. It loads the mesh and bus with an increasing number of *simultaneous* events on a 16×16 grid and counts simulated ticks to full completion (50 trials/level).

| Concurrent events | Fabric (ticks) | Bus (ticks) | Bus/Fabric ratio |
|---|---|---|---|
| 1 | 10.3 | 2 | 0.19× |
| 4 | 17.1 | 8 | 0.47× |
| 16 | 21.8 | 32 | 1.47× |
| 64 | 25.2 | 128 | 5.09× |
| 128 | 26.0 | 256 | 9.84× |

The bus services one hop-step at a time system-wide, so its completion time scales linearly with load (2 ticks × number of events). The mesh only serializes traffic that collides on the same cell at the same hop, so it scales far more slowly. The crossover is around 8–16 simultaneous events on this grid — a regime a single-event benchmark alone would miss.

### `marl_experiment.py` — fault containment (independent vs. shared parameters)

Trains 8 grid-navigation task instances via tabular Q-learning under an equal total compute budget (24,000 episodes either way), either sharing one Q-table (monolithic) or each with its own (modular). After both reach 100% baseline success, 50% of one Q-table's entries are randomly zeroed — the single shared table in the monolithic condition, or one instance's table in the modular condition — and success is re-measured system-wide.

| Condition | Baseline success | Post-failure success (system-wide) |
|---|---|---|
| Monolithic (1 shared table) | 100% | 61.4% |
| Modular (8 independent tables) | 100% | 93.6% |

In the modular condition, only the corrupted instance dropped (to 49%); the other seven stayed at 100%. In the monolithic condition, all eight dropped together, because all eight read from the same table. Caveat carried over from the whitepaper: the shared table benefited from pooled training experience across all 8 instances, a real advantage of sharing that this experiment doesn't erase — the claim is narrowly about fault containment given equal baseline performance, not training efficiency.

## Two real, reproducible results

| Experiment | Finding | Where to verify |
|---|---|---|
| **Fault containment** (independent vs. shared parameters, 8 task instances) | A single fault to a shared parameter set drops system-wide success from 100% to 61.4%. The same fault to one independent unit drops only that unit (100%→49%), leaving the other 7 untouched at 100% (93.6% system-wide) | `marl_experiment.py` + `marl_experiment_results.json`; methodology in whitepaper §4.1 |
| **Concurrency** (decentralized mesh vs. shared bus routing) | A shared bus is faster for one isolated event, but its cost scales linearly with concurrent load. A mesh only serializes actual collisions — by 128 concurrent events, the mesh is ~9.8× faster | `benchmark.py`, `benchmark_concurrency.py`, and their `.json` results; methodology in whitepaper §4.2 |

## Running it

**Primary path — directly from the repo, no download/extraction needed:**

```bash
git clone <this-repo>
cd <this-repo>

# core implementation sanity check
python3 -c "from celos import Fabric; f = Fabric(dim=(8,8)); print(f.pulse())"

# reproduce all three experiments:
python3 benchmark.py
python3 benchmark_concurrency.py
python3 marl_experiment.py
```

**Complete v0.1 package (same code, bundled):**

```bash
unzip cellos-emulator-v0.1-1.zip
cd cellos-emulator
```

Requires Python ≥ 3.9, standard library only — no dependencies.

## Honest caveats

- Both benchmarked comparisons and the fault-containment experiment are **toy-scale**: grid sizes up to 16×16, 8 task instances, tabular (not deep) reinforcement learning.
- All evidence is **software-only** — no custom hardware, FPGA, or silicon exists yet, and no multi-threaded/multi-process runtime was used. All timings are single-process Python behavior on whatever machine ran the script.
- The concurrency benchmark is a **structural/discrete-event model** of contention, not a measurement of true parallel hardware execution.
- The fault-containment experiment tested **one failure mode** (random parameter zeroing at one severity) on **one task**.
- **No independent replication, peer review, or external audit** has occurred.
- **No patents have been filed**; nothing here should be read as an assertion of IP protection.

Full methodology, exact hyperparameters/seeds, and further discussion are in the whitepaper.

## Status

Software-first, early stage. Roadmap: whitepaper (in progress) and v0.1 emulator (in progress, not yet a released package) → provisional patent filing (planned, not yet filed) → FPGA emulation (planned) → custom silicon (planned, contingent on FPGA results).

## Contact

Daysun Simeon — Celos Labs — [celoslab.com](https://celoslab.com)
