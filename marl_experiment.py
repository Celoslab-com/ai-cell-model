"""
Modular vs. Monolithic under failure — a toy MARL experiment
--------------------------------------------------------------
This is the second piece of real evidence for the AI Cell thesis
(alongside the CellOS routing/concurrency benchmark): does giving each
unit its own independent parameters, instead of one shared parameter
set serving all units, actually contain failure the way the theory
claims?

Task: simple tabular Q-learning grid-world navigation, run
independently across N agent "instances" (each instance = its own
random start position, same grid, same goal).

Two conditions, trained under an EQUAL total compute budget
(same total number of training episodes) so the comparison is fair:

  MONOLITHIC — all N instances share ONE Q-table. Every instance's
  experience updates the same shared parameters (a stand-in for a
  monolithic shared-parameter model serving many units).

  MODULAR — each of the N instances has its OWN independent Q-table,
  trained only on its own experience.

After training both to comparable baseline performance, we inject a
REAL failure: randomly corrupt a fraction of Q-table entries.

  - Monolithic: corrupt the one shared table -> every instance is
    affected simultaneously.
  - Modular: corrupt exactly one instance's table -> the other N-1
    instances are completely untouched.

We then re-measure system-wide average success rate for both
conditions. This is a real, reproducible computational experiment —
not a fabricated number. Absolute success rates will depend on the
random seed and hyperparameters below; the meaningful result is the
RELATIVE drop in system-wide performance between the two conditions.
"""

from __future__ import annotations

import random
import statistics
import json
from dataclasses import dataclass, field


GRID_SIZE = 10
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
TRAIN_MAX_STEPS = 60     # generous, so training can explore and converge
EVAL_MAX_STEPS = 20      # tight: max Manhattan distance on this grid is 18,
                         # so evaluation genuinely requires a near-optimal
                         # policy — this is what makes the metric sensitive
                         # to corruption instead of saturating near 100%.
ALPHA = 0.5
GAMMA = 0.9
N_INSTANCES = 8
TRAIN_EPISODES_PER_INSTANCE = 3000  # equal compute budget per instance either way
EVAL_EPISODES = 200
CORRUPTION_FRACTION = 0.5
SEED = 7


def step(pos, action):
    x, y = pos
    dx, dy = ACTIONS[action]
    nx, ny = max(0, min(GRID_SIZE - 1, x + dx)), max(0, min(GRID_SIZE - 1, y + dy))
    new_pos = (nx, ny)
    if new_pos == GOAL:
        return new_pos, 10.0, True
    return new_pos, -1.0, False


def new_q_table():
    return {
        (x, y): [0.0, 0.0, 0.0, 0.0]
        for x in range(GRID_SIZE) for y in range(GRID_SIZE)
    }


def epsilon_greedy(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    qs = q_table[state]
    max_q = max(qs)
    best = [a for a, v in enumerate(qs) if v == max_q]
    return rng.choice(best)


def train_episode(q_table, rng, epsilon):
    pos = (rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
    if pos == GOAL:
        pos = (0, 0)
    for _ in range(TRAIN_MAX_STEPS):
        a = epsilon_greedy(q_table, pos, epsilon, rng)
        new_pos, r, done = step(pos, a)
        best_next = max(q_table[new_pos])
        q_table[pos][a] += ALPHA * (r + GAMMA * best_next - q_table[pos][a])
        pos = new_pos
        if done:
            return True
    return False


def eval_episode(q_table, rng):
    """Tight step budget: only a near-optimal policy can succeed, which
    is what makes this metric sensitive to corruption instead of
    saturating near 100% regardless of damage."""
    pos = (rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
    if pos == GOAL:
        pos = (0, 0)
    for _ in range(EVAL_MAX_STEPS):
        a = epsilon_greedy(q_table, pos, epsilon=0.0, rng=rng)
        pos, r, done = step(pos, a)
        if done:
            return True
    return False


def corrupt(q_table, fraction, rng):
    """Randomly zero out a fraction of (state, action) entries — a
    stand-in for a hardware fault, bad update, or memory corruption."""
    keys = list(q_table.keys())
    n_corrupt = int(len(keys) * fraction)
    for state in rng.sample(keys, n_corrupt):
        q_table[state] = [0.0, 0.0, 0.0, 0.0]


def run_monolithic(rng):
    shared_table = new_q_table()
    epsilon = 0.3
    for _ in range(TRAIN_EPISODES_PER_INSTANCE * N_INSTANCES):
        train_episode(shared_table, rng, epsilon)
        epsilon = max(0.05, epsilon * 0.999)

    baseline = [
        statistics.mean(eval_episode(shared_table, rng) for _ in range(EVAL_EPISODES))
        for _ in range(N_INSTANCES)
    ]

    corrupt(shared_table, CORRUPTION_FRACTION, rng)

    post_failure = [
        statistics.mean(eval_episode(shared_table, rng) for _ in range(EVAL_EPISODES))
        for _ in range(N_INSTANCES)
    ]

    return baseline, post_failure


def run_modular(rng):
    tables = [new_q_table() for _ in range(N_INSTANCES)]
    epsilons = [0.3] * N_INSTANCES

    for _ in range(TRAIN_EPISODES_PER_INSTANCE):
        for i in range(N_INSTANCES):
            train_episode(tables[i], rng, epsilons[i])
            epsilons[i] = max(0.05, epsilons[i] * 0.999)

    baseline = [
        statistics.mean(eval_episode(tables[i], rng) for _ in range(EVAL_EPISODES))
        for i in range(N_INSTANCES)
    ]

    # corrupt exactly ONE instance's independent table
    corrupt(tables[0], CORRUPTION_FRACTION, rng)

    post_failure = [
        statistics.mean(eval_episode(tables[i], rng) for _ in range(EVAL_EPISODES))
        for i in range(N_INSTANCES)
    ]

    return baseline, post_failure


def main():
    rng = random.Random(SEED)

    mono_baseline, mono_post = run_monolithic(rng)
    mod_baseline, mod_post = run_modular(rng)

    results = {
        "config": {
            "n_instances": N_INSTANCES,
            "grid_size": GRID_SIZE,
            "train_episodes_per_instance": TRAIN_EPISODES_PER_INSTANCE,
            "eval_episodes": EVAL_EPISODES,
            "corruption_fraction": CORRUPTION_FRACTION,
            "seed": SEED,
        },
        "monolithic": {
            "per_instance_success_baseline": mono_baseline,
            "per_instance_success_post_failure": mono_post,
            "system_mean_success_baseline": statistics.mean(mono_baseline),
            "system_mean_success_post_failure": statistics.mean(mono_post),
        },
        "modular": {
            "per_instance_success_baseline": mod_baseline,
            "per_instance_success_post_failure": mod_post,
            "system_mean_success_baseline": statistics.mean(mod_baseline),
            "system_mean_success_post_failure": statistics.mean(mod_post),
        },
    }

    print("MONOLITHIC (one shared Q-table across all instances)")
    print(f"  baseline system success rate:     {results['monolithic']['system_mean_success_baseline']*100:.1f}%")
    print(f"  post-failure system success rate: {results['monolithic']['system_mean_success_post_failure']*100:.1f}%")
    print()
    print("MODULAR (independent Q-table per instance)")
    print(f"  baseline system success rate:     {results['modular']['system_mean_success_baseline']*100:.1f}%")
    print(f"  post-failure system success rate: {results['modular']['system_mean_success_post_failure']*100:.1f}%")
    print()
    print(f"Only instance 0 was corrupted in the modular condition.")
    print(f"  instance 0 success rate post-failure: {mod_post[0]*100:.1f}%")
    print(f"  other instances' mean success rate post-failure: {statistics.mean(mod_post[1:])*100:.1f}%")

    with open("marl_experiment_results.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
