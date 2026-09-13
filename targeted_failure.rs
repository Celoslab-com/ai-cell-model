//! Experiment: kill a contiguous cluster of adjacent cells in the center of
//! the grid (not scattered like Chaos Monkey), and test whether the damage
//! actually "stays local" -- do only routes that needed to cross the dead
//! zone get longer, while routes elsewhere are completely unaffected?

use cellos_rs::{random_coord, Fabric, Xorshift};

const GRID: i32 = 32;
const CLUSTER_SIDES: [i32; 3] = [4, 8, 12]; // cluster covers roughly 1.6%, 6.25%, 14% of a 32x32 grid
const TRIALS: usize = 2000;

fn main() {
    println!("Grid: {GRID}x{GRID} ({} cells total)\n", GRID * GRID);
    println!(
        "{:<14}{:<12}{:<24}{:<24}",
        "Cluster", "% of grid", "Routes affected", "Mean overhead (affected only)"
    );

    for &side in &CLUSTER_SIDES {
        let mut fabric = Fabric::new(GRID, GRID);
        fabric.kill_cluster(side);
        let dead_fraction = fabric.dead.len() as f64 / fabric.total_cells() as f64 * 100.0;

        let baseline = Fabric::new(GRID, GRID);
        let mut rng = Xorshift::new(555);

        let mut affected = 0usize;
        let mut unaffected = 0usize;
        let mut overhead_when_affected = Vec::new();
        let mut unreachable = 0usize;
        let mut attempts = 0usize;

        while attempts < TRIALS {
            let s = random_coord(GRID, GRID, &mut rng);
            let t = random_coord(GRID, GRID, &mut rng);
            if s == t || !fabric.is_alive(&s) || !fabric.is_alive(&t) {
                continue;
            }
            attempts += 1;

            let base_path = baseline.shortest_path(s, t).unwrap();
            let base_hops = base_path.len() as f64 - 1.0;

            match fabric.shortest_path(s, t) {
                Some(damaged_path) => {
                    let damaged_hops = damaged_path.len() as f64 - 1.0;
                    if damaged_hops > base_hops {
                        affected += 1;
                        overhead_when_affected.push(damaged_hops - base_hops);
                    } else {
                        unaffected += 1;
                    }
                }
                None => unreachable += 1,
            }
        }

        let pct_affected = affected as f64 / attempts as f64 * 100.0;
        let mean_overhead = if !overhead_when_affected.is_empty() {
            overhead_when_affected.iter().sum::<f64>() / overhead_when_affected.len() as f64
        } else {
            0.0
        };

        println!(
            "{:<14}{:<12}{:<24}{:<24}",
            format!("{side}x{side}"),
            format!("{:.1}%", dead_fraction),
            format!("{:.1}% of {} routes", pct_affected, attempts),
            format!("+{:.2} hops", mean_overhead)
        );

        if unreachable > 0 {
            println!("   ({unreachable} of {attempts} routes had NO surviving path at all -- fabric was disconnected by this cluster)");
        }
        let _ = unaffected;
    }

    println!();
    println!("Honest note: 'affected' means the route's hop count increased because it had to detour");
    println!("around the dead cluster. Routes that don't cross near the cluster are completely unaffected --");
    println!("this is what 'damage stays local' actually means here: a structural property of routes that");
    println!("happen not to need the damaged region, not a claim that the whole fabric is unaffected.");
}
