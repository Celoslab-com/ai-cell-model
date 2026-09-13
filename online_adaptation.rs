//! Experiment 5 — Online Adaptation under Unseen Failures
//!
//! Compares:
//!   1) Static BFS: oracle shortest-path baseline on the post-failure topology.
//!   2) Learned/Frozen: Q-learning policy trained before the failure, frozen during test.
//!   3) Learned/Adaptive: same trained policy, but allowed to update online after failures.
//!
//! The failure topology used during evaluation is different from the training failures.
//! Results are repeated over multiple seeds and written as JSON for reproducibility.

use std::collections::{HashMap, VecDeque};
use std::fs;

const GRID: usize = 16;
const EPISODES: usize = 6000;
const TEST_EPISODES: usize = 1200;
const SEEDS: [u64; 10] = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101];
const ALPHA: f64 = 0.20;
const GAMMA: f64 = 0.95;
const EPS_START: f64 = 0.20;
const EPS_END: f64 = 0.02;
const MAX_STEPS: usize = GRID * GRID * 2;

#[derive(Clone)]
struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed ^ 0x9E3779B97F4A7C15 } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    fn usize(&mut self, n: usize) -> usize { (self.next_u64() as usize) % n }
}

#[derive(Clone)]
struct Grid {
    alive: Vec<bool>,
}
impl Grid {
    fn new() -> Self { Self { alive: vec![true; GRID * GRID] } }
    fn id(r: usize, c: usize) -> usize { r * GRID + c }
    fn rc(id: usize) -> (usize, usize) { (id / GRID, id % GRID) }
    fn neighbors(id: usize) -> [Option<usize>; 4] {
        let (r, c) = Self::rc(id);
        [
            if r > 0 { Some(Self::id(r-1,c)) } else { None },
            if r + 1 < GRID { Some(Self::id(r+1,c)) } else { None },
            if c > 0 { Some(Self::id(r,c-1)) } else { None },
            if c + 1 < GRID { Some(Self::id(r,c+1)) } else { None },
        ]
    }
    fn reset(&mut self) { self.alive.fill(true); }
    fn fail(&mut self, id: usize) {
        if id != 0 && id != GRID * GRID - 1 { self.alive[id] = false; }
    }
}

#[derive(Clone)]
struct Q {
    // state = cell id, action = 0..4 (up/down/left/right/stay)
    q: Vec<[f64; 5]>,
}
impl Q {
    fn new() -> Self { Self { q: vec![[0.0; 5]; GRID * GRID] } }
    fn best(&self, s: usize, grid: &Grid, goal: usize) -> usize {
        let mut best = 4;
        let mut bestv = f64::NEG_INFINITY;
        for a in 0..4 {
            if let Some(n) = Grid::neighbors(s)[a] {
                if grid.alive[n] {
                    let v = self.q[s][a] + 1e-9 * self.progress(n, goal);
                    if v > bestv { bestv = v; best = a; }
                }
            }
        }
        best
    }
    fn progress(&self, id: usize, goal: usize) -> f64 {
        let (r,c) = Grid::rc(id); let (gr,gc) = Grid::rc(goal);
        -((r as i32-gr as i32).abs() + (c as i32-gc as i32).abs()) as f64
    }
    fn choose(&self, s: usize, grid: &Grid, goal: usize, rng: &mut Rng, eps: f64) -> usize {
        if rng.f64() < eps {
            let valid: Vec<usize> = (0..4).filter(|&a| Grid::neighbors(s)[a].map(|n| grid.alive[n]).unwrap_or(false)).collect();
            if valid.is_empty() { 4 } else { valid[rng.usize(valid.len())] }
        } else { self.best(s, grid, goal) }
    }
    fn step(&mut self, s: usize, a: usize, ns: usize, reward: f64, done: bool) {
        let target = if done { reward } else {
            reward + GAMMA * self.q[ns].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        self.q[s][a] += ALPHA * (target - self.q[s][a]);
    }
}

fn apply_training_failures(grid: &mut Grid, rng: &mut Rng) {
    // Training distribution: independent random failures, 5–15% excluding endpoints.
    let p = 0.05 + 0.10 * rng.f64();
    for id in 1..GRID*GRID-1 {
        if rng.f64() < p { grid.fail(id); }
    }
}

fn apply_unseen_cluster(grid: &mut Grid) {
    // Unseen evaluation distribution: a 6x6 central cluster.
    for r in 5..11 {
        for c in 5..11 {
            grid.fail(Grid::id(r,c));
        }
    }
}

fn apply_unseen_random(grid: &mut Grid, rng: &mut Rng) {
    // Separate unseen evaluation distribution: 25% random failures.
    for id in 1..GRID*GRID-1 {
        if rng.f64() < 0.25 { grid.fail(id); }
    }
}

fn bfs(grid: &Grid, start: usize, goal: usize) -> Option<usize> {
    if !grid.alive[start] || !grid.alive[goal] { return None; }
    let mut d = vec![usize::MAX; GRID*GRID];
    let mut q = VecDeque::new();
    d[start] = 0; q.push_back(start);
    while let Some(s) = q.pop_front() {
        if s == goal { return Some(d[s]); }
        for n in Grid::neighbors(s).iter().flatten() {
            if grid.alive[*n] && d[*n] == usize::MAX {
                d[*n] = d[s] + 1; q.push_back(*n);
            }
        }
    }
    None
}

fn rollout(policy: &Q, grid: &Grid, adaptive: bool, rng: &mut Rng, qmut: Option<&mut Q>, goal: usize)
    -> (bool, usize)
{
    let mut state = 0usize;
    let mut visited = HashMap::<usize, usize>::new();
    for step in 0..MAX_STEPS {
        if state == goal { return (true, step); }
        *visited.entry(state).or_insert(0) += 1;
        let eps = if adaptive { 0.05 } else { 0.0 };
        let action = if let Some(q) = qmut.as_ref() {
            q.choose(state, grid, goal, rng, eps)
        } else {
            policy.choose(state, grid, goal, rng, 0.0)
        };
        let next = if action < 4 { Grid::neighbors(state)[action] } else { None };
        let ns = match next {
            Some(n) if grid.alive[n] => n,
            _ => {
                if let Some(q) = qmut {
                    q.step(state, action.min(4), state, -2.0, false);
                }
                continue;
            }
        };
        let loop_penalty = if visited.get(&ns).copied().unwrap_or(0) > 1 { -0.5 } else { 0.0 };
        let done = ns == goal;
        let reward = if done { 20.0 } else { -1.0 + loop_penalty };
        if let Some(q) = qmut {
            q.step(state, action, ns, reward, done);
        }
        state = ns;
        if done { return (true, step + 1); }
    }
    (false, MAX_STEPS)
}

fn train(seed: u64) -> Q {
    let mut rng = Rng::new(seed);
    let mut q = Q::new();
    let goal = GRID*GRID-1;
    for ep in 0..EPISODES {
        let mut grid = Grid::new();
        apply_training_failures(&mut grid, &mut rng);
        let mut state = 0usize;
        let eps = (EPS_START - (EPS_START-EPS_END) * (ep as f64 / EPISODES as f64)).max(EPS_END);
        for _ in 0..MAX_STEPS {
            if state == goal { break; }
            let a = q.choose(state, &grid, goal, &mut rng, eps);
            let next = if a < 4 { Grid::neighbors(state)[a] } else { None };
            match next {
                Some(n) if grid.alive[n] => {
                    let done = n == goal;
                    q.step(state, a, n, if done { 20.0 } else { -1.0 }, done);
                    state = n;
                    if done { break; }
                }
                _ => q.step(state, a, state, -2.0, false),
            }
        }
    }
    q
}

#[derive(Default, Clone)]
struct Metrics {
    success: usize,
    hops: Vec<usize>,
    recovery: Vec<usize>,
}
impl Metrics {
    fn add(&mut self, ok: bool, hops: usize, recovery: usize) {
        if ok { self.success += 1; self.hops.push(hops); self.recovery.push(recovery); }
    }
    fn success_rate(&self, n: usize) -> f64 { 100.0 * self.success as f64 / n as f64 }
    fn mean(v: &[usize]) -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<usize>() as f64 / v.len() as f64 }
    }
}

fn run_condition(seed: u64, learned: &Q, condition: usize) -> Metrics {
    let mut rng = Rng::new(seed ^ 0xABCDEF);
    let mut metrics = Metrics::default();
    let goal = GRID*GRID-1;
    let mut adaptive_q = learned.clone();

    for _ in 0..TEST_EPISODES {
        let mut grid = Grid::new();
        if condition == 0 { apply_unseen_cluster(&mut grid); }
        else { apply_unseen_random(&mut grid, &mut rng); }

        // Simulate an abrupt failure before the episode. Recovery is measured as
        // additional successful steps relative to the oracle BFS route.
        let oracle = bfs(&grid, 0, goal);
        let before = bfs(&Grid::new(), 0, goal).unwrap_or(0);
        let recovery = oracle.map(|h| h.saturating_sub(before)).unwrap_or(MAX_STEPS);

        let (ok, hops) = if condition == 2 {
            rollout(&learned, &grid, true, &mut rng, Some(&mut adaptive_q), goal)
        } else if condition == 1 {
            rollout(&learned, &grid, false, &mut rng, None, goal)
        } else {
            // condition 3 is not used; keep API simple.
            (oracle.is_some(), oracle.unwrap_or(MAX_STEPS))
        };
        metrics.add(ok, hops, recovery);
    }
    metrics
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}
fn std(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    (xs.iter().map(|x| (x-m)*(x-m)).sum::<f64>() / (xs.len()-1) as f64).sqrt()
}
fn ci95(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { 1.96 * std(xs) / (xs.len() as f64).sqrt() }
}

fn json_array(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let mut cluster_bfs = Vec::new();
    let mut cluster_frozen = Vec::new();
    let mut cluster_adapt = Vec::new();
    let mut random_bfs = Vec::new();
    let mut random_frozen = Vec::new();
    let mut random_adapt = Vec::new();

    for &seed in &SEEDS {
        let learned = train(seed);
        let mut cluster_grid = Grid::new(); apply_unseen_cluster(&mut cluster_grid);
        let mut random_grid = Grid::new(); let mut rr = Rng::new(seed ^ 9999); apply_unseen_random(&mut random_grid, &mut rr);

        cluster_bfs.push(if bfs(&cluster_grid,0,GRID*GRID-1).is_some() { 100.0 } else { 0.0 });
        random_bfs.push(if bfs(&random_grid,0,GRID*GRID-1).is_some() { 100.0 } else { 0.0 });

        let cf = run_condition(seed, &learned, 0);
        let ca = {
            let mut rng = Rng::new(seed ^ 0x1234);
            let mut q = learned.clone(); let mut m = Metrics::default();
            for _ in 0..TEST_EPISODES {
                let mut g = Grid::new(); apply_unseen_cluster(&mut g);
                let (ok,h) = rollout(&learned,&g,true,&mut rng,Some(&mut q),GRID*GRID-1);
                m.add(ok,h,0);
            } m
        };
        cluster_frozen.push(cf.success_rate(TEST_EPISODES));
        cluster_adapt.push(ca.success_rate(TEST_EPISODES));

        let rf = run_condition(seed, &learned, 1);
        let ra = {
            let mut rng = Rng::new(seed ^ 0x5678);
            let mut q = learned.clone(); let mut m = Metrics::default();
            for _ in 0..TEST_EPISODES {
                let mut g = Grid::new(); apply_unseen_random(&mut g);
                let (ok,h) = rollout(&learned,&g,true,&mut rng,Some(&mut q),GRID*GRID-1);
                m.add(ok,h,0);
            } m
        };
        random_frozen.push(rf.success_rate(TEST_EPISODES));
        random_adapt.push(ra.success_rate(TEST_EPISODES));
    }

    println!("Experiment 5 — Online Adaptation under Unseen Failures");
    println!("grid={}x{}, training_episodes={}, test_episodes={}, seeds={}", GRID, GRID, EPISODES, TEST_EPISODES, SEEDS.len());
    println!("\nCentral 6x6 unseen cluster:");
    println!("  BFS oracle connectivity: {:.2}% ± {:.2}", mean(&cluster_bfs), ci95(&cluster_bfs));
    println!("  Learned/frozen:          {:.2}% ± {:.2}", mean(&cluster_frozen), ci95(&cluster_frozen));
    println!("  Learned/adaptive:        {:.2}% ± {:.2}", mean(&cluster_adapt), ci95(&cluster_adapt));
    println!("\n25% unseen random failures:");
    println!("  BFS oracle connectivity: {:.2}% ± {:.2}", mean(&random_bfs), ci95(&random_bfs));
    println!("  Learned/frozen:          {:.2}% ± {:.2}", mean(&random_frozen), ci95(&random_frozen));
    println!("  Learned/adaptive:        {:.2}% ± {:.2}", mean(&random_adapt), ci95(&random_adapt));

    let report = format!(
r#"{{
  "experiment": "online_adaptation_unseen_failures",
  "grid": {grid},
  "training_episodes": {train},
  "test_episodes": {test},
  "seeds": {seeds},
  "conditions": ["bfs_oracle", "learned_frozen", "learned_adaptive"],
  "unseen_central_cluster": {{
    "bfs_connectivity_pct": {cb},
    "learned_frozen_success_pct": {cf},
    "learned_adaptive_success_pct": {ca}
  }},
  "unseen_random_25pct": {{
    "bfs_connectivity_pct": {rb},
    "learned_frozen_success_pct": {rf},
    "learned_adaptive_success_pct": {ra}
  }}
}}
"#,
        grid=GRID, train=EPISODES, test=TEST_EPISODES, seeds=SEEDS.len(),
        seeds=json_array(&SEEDS.iter().map(|x| *x as f64).collect::<Vec<_>>()),
        cb=json_array(&cluster_bfs), cf=json_array(&cluster_frozen), ca=json_array(&cluster_adapt),
        rb=json_array(&random_bfs), rf=json_array(&random_frozen), ra=json_array(&random_adapt)
    );
    let _ = fs::write("experiment5_results.json", report);
}
