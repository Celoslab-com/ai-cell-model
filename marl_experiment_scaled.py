"""
Scaled-up fault-containment experiment (v2)
--------------------------------------------
Same experimental design as marl_experiment.py, run at larger scale to test
whether the original result holds beyond the initial toy parameters
(10x10 grid, 8 instances, one 50% corruption severity).

Changes from the original:
  - GRID_SIZE:      10  -> 20   (4x state space)
  - N_INSTANCES:    8   -> 48   (6x)
  - CORRUPTION:     one fixed 50% fraction -> swept over {0.25, 0.5, 0.75}

Training happens ONCE per condition (monolithic / modular). Each corruption
severity is then applied to an independent COPY of the already-trained
table(s), so every severity starts from the identical, fairly-trained
baseline instead of retraining from scratch each time.

Still a software-only, tabular (non-deep) simulation on a toy navigation
task -- larger than the original, but not a production-scale model. See
marl_experiment.py / marl_experiment_results.json for the original,
published-at-smaller-scale numbers, which this does not overwrite.
"""

from __future__ import annotations

import copy
import random
import statistics
import json
import time


GRID_SIZE = 20
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
MAX_MANHATTAN = 2 * (GRID_SIZE - 1)
TRAIN_MAX_STEPS = int(MAX_MANHATTAN * (60 / 18))   # same ratio as original (10x10: 60 vs max-dist 18)
EVAL_MAX_STEPS = int(MAX_MANHATTAN * (20 / 18))    # same tight ratio as original (10x10: 20 vs max-dist 18)
ALPHA = 0.5
GAMMA = 0.9
N_INSTANCES = 48
TRAIN_EPISODES_PER_INSTANCE = 8000   # equal compute budget per instance, both conditions
EVAL_EPISODES = 200
CORRUPTION_FRACTIONS = [0.25, 0.5, 0.75]
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
    keys = list(q_table.keys())
    n_corrupt = int(len(keys) * fraction)
    for state in rng.sample(keys, n_corrupt):
        q_table[state] = [0.0, 0.0, 0.0, 0.0]


def train_monolithic(rng):
    shared_table = new_q_table()
    epsilon = 0.3
    for _ in range(TRAIN_EPISODES_PER_INSTANCE * N_INSTANCES):
        train_episode(shared_table, rng, epsilon)
        epsilon = max(0.05, epsilon * 0.999)
    return shared_table


def train_modular(rng):
    tables = [new_q_table() for _ in range(N_INSTANCES)]
    epsilons = [0.3] * N_INSTANCES
    for _ in range(TRAIN_EPISODES_PER_INSTANCE):
        for i in range(N_INSTANCES):
            train_episode(tables[i], rng, epsilons[i])
            epsilons[i] = max(0.05, epsilons[i] * 0.999)
    return tables


def eval_system(tables_or_table, rng, monolithic: bool):
    if monolithic:
        return [
            statistics.mean(eval_episode(tables_or_table, rng) for _ in range(EVAL_EPISODES))
            for _ in range(N_INSTANCES)
        ]
    return [
        statistics.mean(eval_episode(t, rng) for _ in range(EVAL_EPISODES))
        for t in tables_or_table
    ]


def main():
    t0 = time.time()
    rng = random.Random(SEED)

    print(f"Config: grid={GRID_SIZE}x{GRID_SIZE}, instances={N_INSTANCES}, "
          f"train_episodes/instance={TRAIN_EPISODES_PER_INSTANCE}, "
          f"train_max_steps={TRAIN_MAX_STEPS}, eval_max_steps={EVAL_MAX_STEPS}")

    print("Training monolithic (shared table)...")
    mono_table = train_monolithic(rng)
    mono_baseline = eval_system(mono_table, rng, monolithic=True)
    print(f"  monolithic baseline system success: {statistics.mean(mono_baseline)*100:.1f}%  "
          f"({time.time()-t0:.1f}s elapsed)")

    print("Training modular (independent tables)...")
    mod_tables = train_modular(rng)
    mod_baseline = eval_system(mod_tables, rng, monolithic=False)
    print(f"  modular baseline system success:    {statistics.mean(mod_baseline)*100:.1f}%  "
          f"({time.time()-t0:.1f}s elapsed)")

    results = {
        "config": {
            "grid_size": GRID_SIZE,
            "n_instances": N_INSTANCES,
            "train_episodes_per_instance": TRAIN_EPISODES_PER_INSTANCE,
            "train_max_steps": TRAIN_MAX_STEPS,
            "eval_max_steps": EVAL_MAX_STEPS,
            "eval_episodes": EVAL_EPISODES,
            "corruption_fractions": CORRUPTION_FRACTIONS,
            "seed": SEED,
        },
        "monolithic": {
            "system_mean_success_baseline": statistics.mean(mono_baseline),
            "per_severity": [],
        },
        "modular": {
            "system_mean_success_baseline": statistics.mean(mod_baseline),
            "per_severity": [],
        },
    }

    for frac in CORRUPTION_FRACTIONS:
        print(f"\nSeverity {frac*100:.0f}%:")

        mono_copy = copy.deepcopy(mono_table)
        corrupt(mono_copy, frac, rng)
        mono_post = eval_system(mono_copy, rng, monolithic=True)
        mono_post_mean = statistics.mean(mono_post)
        print(f"  monolithic post-failure system success: {mono_post_mean*100:.1f}%")

        mod_copy = [copy.deepcopy(t) for t in mod_tables]
        corrupt(mod_copy[0], frac, rng)
        mod_post = eval_system(mod_copy, rng, monolithic=False)
        mod_post_mean = statistics.mean(mod_post)
        print(f"  modular post-failure system success:    {mod_post_mean*100:.1f}%  "
              f"(corrupted instance: {mod_post[0]*100:.1f}%, "
              f"others mean: {statistics.mean(mod_post[1:])*100:.1f}%)")

        results["monolithic"]["per_severity"].append({
            "corruption_fraction": frac,
            "system_mean_success_post_failure": mono_post_mean,
            "per_instance_success_post_failure": mono_post,
        })
        results["modular"]["per_severity"].append({
            "corruption_fraction": frac,
            "system_mean_success_post_failure": mod_post_mean,
            "corrupted_instance_success": mod_post[0],
            "other_instances_mean_success": statistics.mean(mod_post[1:]),
            "per_instance_success_post_failure": mod_post,
        })

    with open("marl_experiment_scaled_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("Raw results written to marl_experiment_scaled_results.json")


if __name__ == "__main__":
    main()
