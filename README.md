# The AI Cell Model

**Decentralized compute and modular fault containment as an alternative to monolithic shared-parameter architectures.**

Working paper by Daysun Simeon — Celos Labs. Software-only evidence, not yet peer reviewed.

## The idea

Most AI systems share a resource — a bus, a parameter set — across many tasks. That's efficient when nothing goes wrong, and fragile when something does: a saturated bus or a corrupted shared model degrades everything it serves at once.

The **AI Cell** is a unit that co-locates compute, memory, state, and routing locally, trading some of that shared-resource efficiency for graceful degradation under load and under failure.

## What's in this repo

- **`celos-ai-cell-whitepaper.md`** / **`celos-ai-cell-whitepaper.PDF`** — the full paper
- **`cellos-emulator-v0.1-1.zip`** — Python reference implementation, benchmark harness, fault-injection harness, and raw results for both experiments below

## Two real, reproducible results

| Experiment | Finding |
|---|---|
| **Fault containment** (independent vs. shared parameters, 8 task instances) | A single fault to a shared parameter set drops system-wide success from 100% to 61.4%. The same fault to one independent unit drops only that unit (100%→49%), leaving the other 7 untouched at 100% (93.6% system-wide) |
| **Concurrency** (decentralized mesh vs. shared bus routing) | A shared bus is faster for one isolated event, but its cost scales linearly with concurrent load. A mesh only serializes actual collisions — by 128 concurrent events, the mesh is ~9.8x faster |

Full methodology, honest caveats (toy scale, no custom hardware, no peer review), and exact reproduction steps are in the paper.

## Running it

```bash
unzip cellos-emulator-v0.1-1.zip
cd cellos-emulator
python3 -c "import cellos; fabric = cellos.Fabric(dim=(8,8)); fabric.pulse()"

# reproduce the two headline experiments:
python3 marl_experiment.py
python3 benchmark_concurrency.py
```

Requires Python ≥ 3.9, standard library only — no dependencies.

## Status

Software-first, early stage. Roadmap: whitepaper and v0.1 emulator (done) → FPGA emulation (planned) → custom silicon (planned, contingent on FPGA results).

## Contact

Daysun Simeon — Celos Labs — [celoslab.com](https://celoslab.com)
