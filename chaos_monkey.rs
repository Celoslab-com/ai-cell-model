//! Experiment: kill a random fraction of cells (10%/30%/50%), then measure
//! whether the mesh can still route between surviving cells, compared to a
//! central-hub model where killing the one hub kills all routing.
//!
//! This is a genuinely different question from the earlier fault-containment
//! experiment (which tested corrupted PARAMETERS in a shared vs. independent
//! Q-table). This tests whether the ROUTING SUBSTRATE ITSELF survives losing
//! physical cells -- a new experiment, not a re-run of the old one.

use cellos_rs::{random_coord, CentralHub, Fabric, Xorshift};

const GRID: i32 = 32;
const TRIALS_PER_LEVEL: usize = 500;
const REPS: usize = 10;
const KILL_FRACTIONS: [f64; 3] = [0.10, 0.30, 0.50];

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn main() {
    println!("Grid: {GRID}x{GRID}, {TRIALS_PER_LEVEL} routing attempts per rep, {REPS} reps per kill fraction\n");
    println!(
        "{:<12}{:<26}{:<26}{:<20}",
        "Kill %", "Mesh success rate", "Central-hub success rate", "Mesh path overhead"
    );

    for &frac in &KILL_FRACTIONS {
        let mut mesh_success_rates = Vec::with_capacity(REPS);
        let mut hub_success_rates = Vec::with_capacity(REPS);
        let mut path_overheads = Vec::with_capacity(REPS);

        for rep in 0..REPS {
            let mut rng = Xorshift::new(1000 + rep as u64);

            let mut fabric = Fabric::new(GRID, GRID);
            let mut rng_kill = Xorshift::new(2000 + rep as u64);
            fabric.kill_random(frac, &mut || rng_kill.next_f64());

            // undamaged fabric, same size, to measure baseline path length
            let baseline = Fabric::new(GRID, GRID);

            let mut hub = CentralHub::new(GRID, GRID);
            // the hub itself has the same probability of being among the
            // killed cells as any other cell, at this kill fraction
            hub.hub_alive = !fabric.dead.contains(&hub.hub);

            let mut mesh_successes = 0usize;
            let mut hub_successes = 0usize;
            let mut overhead_samples = Vec::new();
            let mut attempts = 0usize;

            while attempts < TRIALS_PER_LEVEL {
                let s = random_coord(GRID, GRID, &mut rng);
                let t = random_coord(GRID, GRID, &mut rng);
                if s == t || !fabric.is_alive(&s) || !fabric.is_alive(&t) {
                    continue; // only test routing between cells that are actually alive
                }
                attempts += 1;

                if let Some(path) = fabric.shortest_path(s, t) {
                    mesh_successes += 1;
                    if let Some(base_path) = baseline.shortest_path(s, t) {
                        let overhead = (path.len() as f64 - 1.0) - (base_path.len() as f64 - 1.0);
                        overhead_samples.push(overhead);
                    }
                }

                if hub.route(s, t).is_some() {
                    hub_successes += 1;
                }
            }

            mesh_success_rates.push(mesh_successes as f64 / attempts as f64);
            hub_success_rates.push(hub_successes as f64 / attempts as f64);
            if !overhead_samples.is_empty() {
                path_overheads.push(mean(&overhead_samples));
            }
        }

        let mesh_m = mean(&mesh_success_rates) * 100.0;
        let mesh_s = std(&mesh_success_rates) * 100.0;
        let hub_m = mean(&hub_success_rates) * 100.0;
        let overhead_m = if !path_overheads.is_empty() { mean(&path_overheads) } else { 0.0 };

        println!(
            "{:<12}{:<26}{:<26}{:<20}",
            format!("{:.0}%", frac * 100.0),
            format!("{:.1}% ± {:.1}%", mesh_m, mesh_s),
            format!("{:.0}% (hub {})", hub_m, if hub_m > 0.0 { "survived" } else { "killed" }),
            format!("+{:.2} hops", overhead_m)
        );
    }

    println!();
    println!("Honest note: 'central-hub success rate' here is deterministic per rep (100% if the hub");
    println!("happened to survive being randomly killed, 0% if it didn't) -- it is not a graceful-degradation");
    println!("model, it is a single-point-of-failure model, which is exactly the property being tested.");
}
