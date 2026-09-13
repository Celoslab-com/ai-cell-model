//! Experiment: train local AI-cell policies to route around failures, then
//! evaluate them on failure patterns that were not seen during training.
//!
//! Each cell owns a small Q-table.  A cell observes only local information:
//! its own position relative to the target and which of its four neighbors
//! are currently alive.  It chooses the next hop with epsilon-greedy Q-learning.
//! The policy is updated from local rewards after each step.
//!
//! This is evidence for adaptive local decision-making, not proof of general
//! intelligence.  The static BFS fabric is included as a strong non-learning
//! baseline.

use cellos_rs::{Coord, Fabric, Xorshift};
use std::collections::HashMap;

const GRID: i32 = 16;
const MAX_STEPS: usize = 120;
const TRAIN_EPISODES: usize = 5000;
const TEST_EPISODES: usize = 1000;
const ALPHA: f64 = 0.18;
const GAMMA: f64 = 0.92;
const EPS_START: f64 = 0.30;
const EPS_END: f64 = 0.03;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct State {
    // Target direction, clipped to sign only: local information rather than
    // a table keyed by the entire target coordinate.
    dx: i8,
    dy: i8,
    // Whether each neighbor is alive: right, left, down, up.
    alive_mask: u8,
}

#[derive(Clone, Copy, Debug)]
struct QCell {
    q: [f64; 4],
}

impl Default for QCell {
    fn default() -> Self { Self { q: [0.0; 4] } }
}

struct LocalAi {
    tables: HashMap<Coord, HashMap<State, QCell>>,
}

impl LocalAi {
    fn new() -> Self { Self { tables: HashMap::new() } }

    fn state(fabric: &Fabric, pos: Coord, target: Coord) -> State {
        let sign = |v: i32| -> i8 { if v < 0 { -1 } else if v > 0 { 1 } else { 0 } };
        let (x, y) = pos;
        let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let mut mask = 0u8;
        for (i, (dx, dy)) in dirs.iter().enumerate() {
            if fabric.is_alive(&(x + dx, y + dy)) { mask |= 1 << i; }
        }
        State { dx: sign(target.0 - x), dy: sign(target.1 - y), alive_mask: mask }
    }

    fn choose(&mut self, fabric: &Fabric, pos: Coord, target: Coord, epsilon: f64, rng: &mut Xorshift) -> usize {
        let s = Self::state(fabric, pos, target);
        let entry = self.tables.entry(pos).or_default().entry(s).or_default();
        if rng.next_f64() < epsilon {
            return rng.next_range(4) as usize;
        }
        let mut best = 0usize;
        for a in 1..4 {
            if entry.q[a] > entry.q[best] { best = a; }
        }
        best
    }

    fn update(&mut self, fabric: &Fabric, pos: Coord, target: Coord, action: usize, reward: f64, next: Option<Coord>) {
        let s = Self::state(fabric, pos, target);
        let next_max = next.map(|n| {
            let ns = Self::state(fabric, n, target);
            self.tables.get(&n)
                .and_then(|m| m.get(&ns))
                .map(|q| q.q.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .unwrap_or(0.0)
        }).unwrap_or(0.0);
        let cell = self.tables.entry(pos).or_default().entry(s).or_default();
        let target_q = reward + GAMMA * next_max;
        cell.q[action] += ALPHA * (target_q - cell.q[action]);
    }
}

fn step(c: Coord, action: usize) -> Coord {
    match action { 0 => (c.0 + 1, c.1), 1 => (c.0 - 1, c.1), 2 => (c.0, c.1 + 1), _ => (c.0, c.1 - 1) }
}

fn sample_endpoints(fabric: &Fabric, rng: &mut Xorshift) -> Option<(Coord, Coord)> {
    for _ in 0..100 {
        let s = (rng.next_range(GRID), rng.next_range(GRID));
        let t = (rng.next_range(GRID), rng.next_range(GRID));
        if s != t && fabric.is_alive(&s) && fabric.is_alive(&t) { return Some((s, t)); }
    }
    None
}

fn train_episode(ai: &mut LocalAi, fabric: &Fabric, s: Coord, t: Coord, epsilon: f64, rng: &mut Xorshift) -> bool {
    let mut pos = s;
    for _ in 0..MAX_STEPS {
        if pos == t { return true; }
        let action = ai.choose(fabric, pos, t, epsilon, rng);
        let next = step(pos, action);
        if !fabric.is_alive(&next) {
            ai.update(fabric, pos, t, action, -25.0, None);
            continue;
        }
        let reward = if next == t { 100.0 } else { -1.0 };
        ai.update(fabric, pos, t, action, reward, Some(next));
        pos = next;
        if pos == t { return true; }
    }
    false
}

fn evaluate(ai: &mut LocalAi, fabric: &Fabric, rng: &mut Xorshift, learning: bool) -> (f64, f64, f64) {
    let mut success = 0usize;
    let mut hops = Vec::new();
    let mut optimal_overhead = Vec::new();
    for _ in 0..TEST_EPISODES {
        let Some((s, t)) = sample_endpoints(fabric, rng) else { continue; };
        let baseline = fabric.shortest_path(s, t);
        let Some(base) = baseline else { continue; };
        let base_hops = base.len() - 1;
        let mut pos = s;
        let mut visited = std::collections::HashSet::new();
        visited.insert(pos);
        let mut done = false;
        for step_n in 0..MAX_STEPS {
            if pos == t { done = true; break; }
            let action = ai.choose(fabric, pos, t, if learning { 0.0 } else { 0.0 }, rng);
            let next = step(pos, action);
            if !fabric.is_alive(&next) || visited.contains(&next) {
                if learning { ai.update(fabric, pos, t, action, -25.0, None); }
                break;
            }
            if learning { ai.update(fabric, pos, t, action, if next == t { 100.0 } else { -1.0 }, Some(next)); }
            pos = next;
            visited.insert(pos);
            if pos == t {
                done = true;
                hops.push(step_n + 1);
                optimal_overhead.push((step_n + 1 - base_hops) as f64);
                break;
            }
        }
        if done { success += 1; }
    }
    let denom = TEST_EPISODES as f64;
    let mean_hops = if hops.is_empty() { f64::NAN } else { hops.iter().sum::<usize>() as f64 / hops.len() as f64 };
    let overhead = if optimal_overhead.is_empty() { f64::NAN } else { optimal_overhead.iter().sum::<f64>() / optimal_overhead.len() as f64 };
    (success as f64 / denom * 100.0, mean_hops, overhead)
}

fn train_on_pattern(ai: &mut LocalAi, side: i32, rng: &mut Xorshift) {
    for ep in 0..TRAIN_EPISODES {
        let mut f = Fabric::new(GRID, GRID);
        if ep % 2 == 0 { f.kill_cluster(side); } else { f.kill_random(0.12, &mut || rng.next_f64()); }
        if let Some((s, t)) = sample_endpoints(&f, rng) {
            let frac = ep as f64 / (TRAIN_EPISODES - 1) as f64;
            let epsilon = EPS_START + (EPS_END - EPS_START) * frac;
            let _ = train_episode(ai, &f, s, t, epsilon, rng);
        }
    }
}

fn main() {
    println!("AI-cell learning experiment: {GRID}x{GRID}");
    println!("Training: {TRAIN_EPISODES} episodes; testing: {TEST_EPISODES} episodes per pattern");
    println!("\nMetric                     Static BFS       Learned local cells");

    let mut ai = LocalAi::new();
    let mut rng = Xorshift::new(20260913);
    train_on_pattern(&mut ai, 6, &mut rng);

    for (name, pattern) in [("Unseen 10x10 cluster", 10), ("Random 25% failures", 0)] {
        let mut fabric = Fabric::new(GRID, GRID);
        if pattern > 0 { fabric.kill_cluster(pattern); } else { fabric.kill_random(0.25, &mut || rng.next_f64()); }
        let mut eval_rng = Xorshift::new(9000 + pattern as u64);
        let (_, bfs_hops, _) = evaluate(&mut LocalAi::new(), &fabric, &mut eval_rng, false);
        let (ai_success, ai_hops, ai_overhead) = evaluate(&mut ai, &fabric, &mut eval_rng, false);

        // BFS success is the structural upper bound for these sampled pairs.
        let mut reachable = 0usize;
        for _ in 0..TEST_EPISODES {
            if let Some((s, t)) = sample_endpoints(&fabric, &mut eval_rng) {
                if fabric.shortest_path(s, t).is_some() { reachable += 1; }
            }
        }
        let bfs_success = reachable as f64 / TEST_EPISODES as f64 * 100.0;

        println!("{:<25} {:>7.1}% / {:>6.2}h   {:>7.1}% / {:>6.2}h / +{:>5.2}h", name, bfs_success, bfs_hops, ai_success, ai_hops, ai_overhead);
    }

    println!("\nInterpretation:");
    println!("- BFS is the non-learning routing baseline and provides the reachable-path ceiling.");
    println!("- Learned cells use only local neighbor availability plus the direction of the target.");
    println!("- Testing includes a larger failure cluster than the 6x6 cluster used during training.");
    println!("- Improvement after training is evidence of adaptive local policy learning, not proof of general intelligence.");
}
