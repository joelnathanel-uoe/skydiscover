//! CLI wrapper around the `jss-instance-gen` library.
//!
//! Usage:
//!     jss-instance-gen <NUM_INSTANCES> <SCENARIO> [OUT_DIR]
//!
//! See the README for details.

use jss_instance_gen::{generate_instances, Scenario};
use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;

const USAGE: &str = "\
jss-instance-gen — generate TIG job_scheduling (FJSP) benchmark instances

USAGE:
    jss-instance-gen <NUM_INSTANCES> <SCENARIO> [OUT_DIR]

ARGS:
    <NUM_INSTANCES>   How many instances to generate (e.g. 100)
    <SCENARIO>        One of: flow_shop | hybrid_flow_shop | job_shop | fjsp_medium | fjsp_high
    [OUT_DIR]         Output directory (default: ./instances)

EXAMPLE:
    jss-instance-gen 100 fjsp_medium ./benchmark
";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") || args.len() < 3 {
        print!("{}", USAGE);
        exit(if args.len() < 3 { 1 } else { 0 });
    }

    let num_instances: usize = match args[1].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: NUM_INSTANCES must be a non-negative integer, got '{}'\n", args[1]);
            print!("{}", USAGE);
            exit(1);
        }
    };

    let scenario = match Scenario::from_str(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}\n", e);
            print!("{}", USAGE);
            exit(1);
        }
    };

    let out_dir = PathBuf::from(args.get(3).map(|s| s.as_str()).unwrap_or("instances"));

    match generate_instances(num_instances, scenario, &out_dir, None) {
        Ok(paths) => {
            println!(
                "Wrote {} instance(s) ({}) to {}",
                paths.len(),
                scenario,
                out_dir.display()
            );
            println!("Manifest: {}", out_dir.join("manifest.csv").display());
        }
        Err(e) => {
            eprintln!("Error writing instances: {}", e);
            exit(1);
        }
    }
}
