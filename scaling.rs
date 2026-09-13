//! Experiment: does routing computation overhead scale linearly (or worse)
//! as the fabric grows from 4x4 up to 128x128?
//!
//! Real wall-clock measurement, real Rust code, run on this machine.
//! Absolute nanosecond timings are machine-specific; the meaningful
//! signal is the growth trend across grid sizes, not the raw numbers.

use cellos_rs::{random_coord, Fabric, Xorshift};
use std::time::Instant;

fn main() {
    let sizes = [4, 8, 16, 32, 64, 128];
    let trials_per_size = 300;

    println!("{:<10}{:<16}{:<16}{:<12}", "Grid", "Mean us/route", "Std us/route", "Mean hops");

    let mut results = Vec::new();

    for &n in &sizes {
        let fabric = Fabric::new(n, n);
        let mut rng = Xorshift::new(42);
        let mut timings_us = Vec::with_capacity(trials_per_size);
        let mut hop_counts = Vec::with_capacity(trials_per_size);

        for _ in 0..trials_per_size {
            let s = random_coord(n, n, &mut rng);
            let t = random_coord(n, n, &mut rng);
            if s == t {
                continue;
            }
            let start = Instant::now();
            let path = fabric.shortest_path(s, t);
            let elapsed = start.elapsed().as_nanos() as f64 / 1000.0; // -> microseconds
            timings_us.push(elapsed);
            if let Some(p) = path {
                hop_counts.push((p.len() - 1) as f64);
            }
        }

        let mean = timings_us.iter().sum::<f64>() / timings_us.len() as f64;
        let variance = timings_us.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / timings_us.len() as f64;
        let std = variance.sqrt();
        let mean_hops = hop_counts.iter().sum::<f64>() / hop_counts.len() as f64;

        println!("{:<10}{:<16.3}{:<16.3}{:<12.2}", format!("{n}x{n}"), mean, std, mean_hops);
        results.push((n, mean, std, mean_hops));
    }

    // Report the scaling exponent between smallest and largest grid, as a
    // simple check on whether cost grows roughly linearly with cell count
    // (cell count grows as n^2, so "linear in cell count" means time should
    // grow roughly with n^2 too, not faster).
    let (n0, t0, _, _) = results[0];
    let (n1, t1, _, _) = results[results.len() - 1];
    let cell_ratio = (n1 * n1) as f64 / (n0 * n0) as f64;
    let time_ratio = t1 / t0;
    println!();
    println!(
        "Cell count grew {:.0}x ({n0}x{n0} -> {n1}x{n1}); routing time grew {:.1}x.",
        cell_ratio, time_ratio
    );
    if time_ratio < cell_ratio * 1.5 {
        println!("Growth is roughly in line with (or better than) cell-count growth -- no evidence of exponential blowup in this range.");
    } else {
        println!("Growth exceeds cell-count growth -- worth investigating before claiming this scales cleanly.");
    }
}
