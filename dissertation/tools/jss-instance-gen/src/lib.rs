//! Standalone generator for TIG `job_scheduling` (Flexible Job Shop, FJSP) instances.
//!
//! This crate is a self-contained extraction of the instance generator from the TIG
//! monorepo (`tig-challenges/src/job_scheduling/`). It has no dependency on the rest of
//! the repo — only `rand` and `rand_distr`.
//!
//! IMPORTANT: `generate_instance` below is copied VERBATIM from the TIG challenge, including
//! the exact RNG seeding chain and the exact order of every `rng` call. The generated
//! instance is fully determined by that sequence, so the instances this crate produces are
//! bit-identical to the ones TIG generates for the same seed and scenario. Do not reorder or
//! "clean up" the body — doing so silently changes which instances come out.

use rand::{
    distributions::Distribution,
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use rand_distr::Normal;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Scenario (shop type) — lifted from job_scheduling/scenarios.rs
// ---------------------------------------------------------------------------

pub struct ScenarioConfig {
    pub avg_op_flexibility: f32,
    pub reentrance_level: f32,
    pub flow_structure: f32,
    pub product_mix_ratio: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Scenario {
    FLOW_SHOP,
    HYBRID_FLOW_SHOP,
    JOB_SHOP,
    FJSP_MEDIUM,
    FJSP_HIGH,
}

impl From<Scenario> for ScenarioConfig {
    fn from(scenario: Scenario) -> Self {
        match scenario {
            Scenario::FLOW_SHOP => ScenarioConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Scenario::HYBRID_FLOW_SHOP => ScenarioConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Scenario::JOB_SHOP => ScenarioConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.0,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Scenario::FJSP_MEDIUM => ScenarioConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Scenario::FJSP_HIGH => ScenarioConfig {
                avg_op_flexibility: 10.0,
                reentrance_level: 0.0,
                flow_structure: 1.0,
                product_mix_ratio: 1.0,
            },
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::FLOW_SHOP => write!(f, "flow_shop"),
            Scenario::HYBRID_FLOW_SHOP => write!(f, "hybrid_flow_shop"),
            Scenario::JOB_SHOP => write!(f, "job_shop"),
            Scenario::FJSP_MEDIUM => write!(f, "fjsp_medium"),
            Scenario::FJSP_HIGH => write!(f, "fjsp_high"),
        }
    }
}

impl FromStr for Scenario {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "flow_shop" => Ok(Scenario::FLOW_SHOP),
            "hybrid_flow_shop" => Ok(Scenario::HYBRID_FLOW_SHOP),
            "job_shop" => Ok(Scenario::JOB_SHOP),
            "fjsp_medium" => Ok(Scenario::FJSP_MEDIUM),
            "fjsp_high" => Ok(Scenario::FJSP_HIGH),
            _ => Err(format!(
                "Invalid scenario '{}'. Expected one of: flow_shop, hybrid_flow_shop, job_shop, fjsp_medium, fjsp_high",
                s
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Challenge (instance) — the fields needed to describe & serialize an FJSP instance
// ---------------------------------------------------------------------------

// due-date tightness factor g (moderate regime):
// d_j = DUE_DATE_TIGHTNESS * FF * avg_raw_processing_j
fn due_date_tightness() -> f64 {
    std::env::var("JSS_G").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5)
}

#[derive(Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_jobs: usize,
    pub num_machines: usize,
    pub num_operations: usize,
    pub jobs_per_product: Vec<usize>,
    // each product has a sequence of operations, and each operation has a map of
    // eligible machines to processing times
    pub product_processing_times: Vec<Vec<HashMap<usize, u32>>>,
    // per-job (flattened job order) total-weighted-tardiness data
    pub due_dates: Vec<u32>,
    pub weights: Vec<u32>,
}

impl Challenge {
    /// VERBATIM copy of `Challenge::generate_instance` from
    /// `tig-challenges/src/job_scheduling/mod.rs`, with the signature adapted to take a
    /// `Scenario` directly (instead of a `Track`). The body — including the RNG seeding
    /// and every `rng` call — is unchanged. This is the **total-weighted-tardiness**
    /// version of the challenge: after the FJSP structure is built it derives a due date
    /// and a weight for every job. Returns `Err` only if the internal FIFO pass stalls
    /// (does not happen for well-formed instances).
    pub fn generate_instance(seed: &[u8; 32], scenario: Scenario) -> Result<Self, String> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed.clone()).r#gen());
        let ScenarioConfig {
            avg_op_flexibility,
            reentrance_level,
            flow_structure,
            product_mix_ratio,
        } = scenario.into();
        let n_jobs = 50;
        let n_machines = n_jobs / 2 + 5;
        let n_op_types = n_jobs / 2 + 5;
        let n_products = 1.max((product_mix_ratio * n_jobs as f32) as usize);
        let n_routes = 1.max((flow_structure * n_jobs as f32) as usize);
        let min_eligible_machines = 1;
        let flexibility_std_dev = 0.5;
        let base_proc_time_min = 1;
        let base_proc_time_max = 200;
        let min_speed_factor: f32 =
            std::env::var("JSS_SPEED_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.8);
        let max_speed_factor: f32 =
            std::env::var("JSS_SPEED_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1.2);

        // random product for each job, only keep products that have at least one job
        let mut map = HashMap::new();
        let jobs_per_product = (0..n_jobs).fold(Vec::new(), |mut acc, _| {
            let map_len = map.len();
            let product = *map
                .entry(rng.gen_range(0..n_products))
                .or_insert_with(|| map_len);
            if product >= acc.len() {
                acc.push(0);
            }
            acc[product] += 1;
            acc
        });
        // actual number of products (some products may have zero jobs)
        let n_products = jobs_per_product.len();

        // random route for each product, only keep routes that are used
        let mut map = HashMap::new();
        let product_route = (0..n_products)
            .map(|_| {
                let map_len = map.len();
                *map.entry(rng.gen_range(0..n_routes))
                    .or_insert_with(|| map_len)
            })
            .collect::<Vec<usize>>();
        // actual number of routes
        let n_routes = map.len();

        // generate operation sequence for each route
        let routes = (0..n_routes)
            .map(|_| {
                let seq_len = n_op_types;
                let mut base_sequence: Vec<usize> = (0..n_op_types).collect();
                let mut steps = Vec::new();

                // randomly build op sequence
                for _ in 0..seq_len {
                    let next_op_idx = if rng.r#gen::<f32>() < flow_structure {
                        // Job Shop Logic: Random permutation
                        rng.gen_range(0..base_sequence.len())
                    } else {
                        // Scenario Shop Logic: Pick next sequential op
                        0
                    };

                    let op_id = base_sequence.remove(next_op_idx);
                    steps.push(op_id);
                }

                for step_idx in (2..steps.len()).rev() {
                    // Reentrance Logic
                    if rng.r#gen::<f32>() < reentrance_level {
                        // assuming reentrance_level of 0.1
                        let op_id = steps[rng.gen_range(0..step_idx - 1)];
                        steps.insert(step_idx, op_id);
                    }
                }

                steps
            })
            .collect::<Vec<Vec<usize>>>();

        // generate machine eligibility and base processing time for each operation
        let normal = Normal::new(avg_op_flexibility, flexibility_std_dev).unwrap();
        let all_machines = (0..n_machines).collect::<HashSet<usize>>();
        let op_eligible_machines = (0..n_op_types)
            .map(|i| {
                if avg_op_flexibility as usize >= n_machines {
                    (0..n_machines).collect::<Vec<usize>>()
                } else {
                    let mut eligible = HashSet::<usize>::from([if i < n_machines {
                        i
                    } else {
                        rng.gen_range(0..n_machines)
                    }]);
                    if avg_op_flexibility > 1.0 {
                        let target_flex = min_eligible_machines
                            .max(normal.sample(&mut rng) as usize)
                            .min(n_machines);
                        let mut remaining = all_machines
                            .difference(&eligible)
                            .cloned()
                            .collect::<Vec<usize>>();
                        remaining.sort_unstable();
                        let num_to_add = (target_flex - 1).min(remaining.len());
                        for j in 0..num_to_add {
                            let idx = rng.gen_range(j..remaining.len());
                            remaining.swap(j, idx);
                        }
                        eligible.extend(remaining[..num_to_add].iter().cloned());
                    }
                    let mut eligible = eligible.into_iter().collect::<Vec<usize>>();
                    eligible.sort_unstable();
                    eligible
                }
            })
            .collect::<Vec<_>>();
        let base_proc_times = (0..n_op_types)
            .map(|_| rng.gen_range(base_proc_time_min..=base_proc_time_max))
            .collect::<Vec<u32>>();

        // generate processing times for each product according to its route
        let product_processing_times = product_route
            .iter()
            .map(|&r_idx| {
                let route = &routes[r_idx];
                route
                    .iter()
                    .map(|&op_id| {
                        let machines = &op_eligible_machines[op_id];
                        let base_time = base_proc_times[op_id];
                        machines
                            .iter()
                            .map(|&m_id| {
                                (
                                    m_id,
                                    1.max(
                                        (base_time as f32
                                            * (min_speed_factor
                                                + (max_speed_factor - min_speed_factor)
                                                    * rng.r#gen::<f32>()))
                                            as u32,
                                    ),
                                )
                            })
                            .collect::<HashMap<usize, u32>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // flatten jobs into (product) order, matching the evaluator and baselines
        let job_products = jobs_per_product
            .iter()
            .enumerate()
            .flat_map(|(product, &count)| std::iter::repeat(product).take(count))
            .collect::<Vec<usize>>();

        // average raw processing time per product: sum over its operations of the
        // mean processing time across that operation's eligible machines
        let avg_raw_processing = product_processing_times
            .iter()
            .map(|ops| {
                ops.iter()
                    .map(|op| op.values().sum::<u32>() as f64 / op.len() as f64)
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();

        // build the challenge with placeholder due dates / weights so the FIFO pass can read it
        let mut challenge = Challenge {
            seed: seed.clone(),
            num_jobs: n_jobs,
            num_machines: n_machines,
            num_operations: n_op_types,
            jobs_per_product,
            product_processing_times,
            due_dates: Vec::new(),
            weights: Vec::new(),
        };

        // single-pass FIFO schedule to estimate the flow factor FF (>= 1)
        let (fifo_completion, fifo_processing) = fifo_pass(&challenge)?;
        let ff = {
            let mut sum = 0.0f64;
            for j in 0..n_jobs {
                if fifo_processing[j] > 0.0 {
                    sum += fifo_completion[j] as f64 / fifo_processing[j];
                }
            }
            (sum / n_jobs as f64).max(1.0)
        };

        // due date d_j = g * FF * avg_raw_processing_{product(j)}; weight w_j ~ Uniform{1,2,3}
        challenge.due_dates = (0..n_jobs)
            .map(|j| {
                let p = job_products[j];
                {
                    let d = due_date_tightness() * ff * avg_raw_processing[p];
                    let d = if std::env::var("JSS_DUE_FLOOR").is_ok() { d.floor() } else { d.round() };
                    d.max(1.0) as u32
                }
            })
            .collect();
        challenge.weights = (0..n_jobs).map(|_| rng.gen_range(1..=3)).collect();

        Ok(challenge)
    }

    /// Render this instance in the standard FJSP benchmark text format
    /// (Brandimarte / Hurink / Fattahi):
    ///
    /// ```text
    /// <num_jobs> <num_machines>
    /// # then one line per job:
    /// <num_ops>  <n_elig> <mach> <time> <mach> <time> ...  <n_elig> <mach> <time> ...
    /// ```
    ///
    /// Per job: the operation count, then for each operation the number of eligible
    /// machines followed by `(machine, processing_time)` pairs. Machine ids are
    /// **1-indexed**, which is the convention in published FJSP datasets.
    ///
    /// Jobs are emitted in the same flattened product order the TIG evaluator uses
    /// (all jobs of product 0, then product 1, ...), so a job's line index here matches
    /// its job index in TIG.
    ///
    /// Because the TIG challenge optimises **total weighted tardiness**, two extra lines
    /// are appended after the standard FJSP body — the per-job due dates and weights, in
    /// the same job order:
    ///
    /// ```text
    /// DUE_DATES: d_0 d_1 ... d_{n-1}
    /// WEIGHTS:   w_0 w_1 ... w_{n-1}
    /// ```
    ///
    /// A plain FJSP/makespan parser reads the header + `num_jobs` job lines and simply
    /// ignores the two trailing labelled lines, so the file stays compatible with standard
    /// FJSP tooling while still carrying the tardiness data.
    pub fn to_fjsp(&self) -> String {
        let mut out = format!("{} {}\n", self.num_jobs, self.num_machines);
        for (product, &count) in self.jobs_per_product.iter().enumerate() {
            let ops = &self.product_processing_times[product];
            for _ in 0..count {
                let mut line = format!("{}", ops.len());
                for op in ops {
                    line.push_str(&format!(" {}", op.len()));
                    // sort by machine id for stable, deterministic output
                    let mut pairs: Vec<(&usize, &u32)> = op.iter().collect();
                    pairs.sort_by_key(|(m, _)| **m);
                    for (m, t) in pairs {
                        line.push_str(&format!(" {} {}", m + 1, t));
                    }
                }
                out.push_str(&line);
                out.push('\n');
            }
        }
        let due: Vec<String> = self.due_dates.iter().map(|d| d.to_string()).collect();
        let wts: Vec<String> = self.weights.iter().map(|w| w.to_string()).collect();
        out.push_str(&format!("DUE_DATES: {}\n", due.join(" ")));
        out.push_str(&format!("WEIGHTS: {}\n", wts.join(" ")));
        out
    }
}

// Earliest finish time of an operation across its eligible machines, given the current
// time and machine availability. Verbatim from the challenge baseline.
fn earliest_end_time(
    time: u32,
    machine_available_time: &[u32],
    operation: &HashMap<usize, u32>,
) -> u32 {
    let mut earliest_end = u32::MAX;
    for (&machine_id, &proc_time) in operation.iter() {
        let start = time.max(machine_available_time[machine_id]);
        let end = start + proc_time;
        if end < earliest_end {
            earliest_end = end;
        }
    }
    earliest_end
}

/// VERBATIM copy of `fifo_pass` from `dispatching_rules.rs` (error type changed to `String`).
/// Single deterministic FIFO list-scheduling pass used during instance generation to estimate
/// the flow factor. Returns each job's completion time and its realized total processing time
/// (both in flattened job order). Tie-break is lowest job index (first-in-first-out).
/// Uses no RNG, so it does not affect instance reproducibility.
pub fn fifo_pass(challenge: &Challenge) -> Result<(Vec<u32>, Vec<f64>), String> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (product, count) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..*count {
            job_products.push(product);
        }
    }

    let job_ops_len = job_products
        .iter()
        .map(|&p| challenge.product_processing_times[p].len())
        .collect::<Vec<usize>>();

    let mut job_next_op_idx = vec![0usize; num_jobs];
    let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_available_time = vec![0u32; num_machines];
    let mut job_processing_time = vec![0.0f64; num_jobs];

    let mut remaining_ops = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;

    while remaining_ops > 0 {
        let mut available_machines = (0..num_machines)
            .filter(|&m| machine_available_time[m] <= time)
            .collect::<Vec<usize>>();
        available_machines.sort_unstable();

        let mut scheduled_any = false;
        for &machine in available_machines.iter() {
            // FIFO: lowest-index ready job whose next operation runs earliest on this machine
            let mut chosen: Option<usize> = None;
            for job in 0..num_jobs {
                if job_next_op_idx[job] >= job_ops_len[job] {
                    continue;
                }
                if job_ready_time[job] > time {
                    continue;
                }
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let proc_time = match op_times.get(&machine) {
                    Some(&value) => value,
                    None => continue,
                };
                let earliest_end = earliest_end_time(time, &machine_available_time, op_times);
                let machine_end = time.max(machine_available_time[machine]) + proc_time;
                if machine_end != earliest_end {
                    continue;
                }
                chosen = Some(job);
                break;
            }

            if let Some(job) = chosen {
                let product = job_products[job];
                let op_idx = job_next_op_idx[job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let proc_time = op_times[&machine];
                let start_time = time.max(machine_available_time[machine]);
                let end_time = start_time + proc_time;

                job_next_op_idx[job] += 1;
                job_ready_time[job] = end_time;
                machine_available_time[machine] = end_time;
                job_processing_time[job] += proc_time as f64;
                remaining_ops -= 1;
                scheduled_any = true;
            }
        }

        if remaining_ops == 0 {
            break;
        }

        let mut next_time: Option<u32> = None;
        for &t in machine_available_time.iter() {
            if t > time {
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }
        for job in 0..num_jobs {
            if job_next_op_idx[job] < job_ops_len[job] && job_ready_time[job] > time {
                let t = job_ready_time[job];
                next_time = Some(next_time.map_or(t, |best| best.min(t)));
            }
        }

        time = next_time.ok_or_else(|| {
            if scheduled_any {
                "No next event time found while operations remain unscheduled".to_string()
            } else {
                "No schedulable operations remain; FIFO pass stalled".to_string()
            }
        })?;
    }

    Ok((job_ready_time, job_processing_time))
}

/// Derive a deterministic 32-byte seed from an instance index, so that a whole benchmark
/// set is reproducible from just `(num_instances, scenario)`. If `base_seed` is provided,
/// the index is mixed into it; otherwise a zero base is used.
pub fn seed_for_index(index: u64, base_seed: Option<[u8; 32]>) -> [u8; 32] {
    let mut seed = base_seed.unwrap_or([0u8; 32]);
    // XOR the little-endian index into the first 8 bytes (so an explicit base seed is honored)
    let idx_bytes = index.to_le_bytes();
    for i in 0..8 {
        seed[i] ^= idx_bytes[i];
    }
    seed
}

/// Generate `num_instances` instances of `scenario` and write them as standard FJSP
/// `.txt` files into `out_dir`. Files are named `instance_000.txt`, `instance_001.txt`, ...
/// (zero-padded to at least 3 digits). Also writes a `manifest.csv` summarising the set.
///
/// Returns the list of file paths written (instance files only).
pub fn generate_instances(
    num_instances: usize,
    scenario: Scenario,
    out_dir: &Path,
    base_seed: Option<[u8; 32]>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(out_dir)?;

    let width = (num_instances.max(1) - 1).to_string().len().max(3);
    let mut paths = Vec::with_capacity(num_instances);
    let mut manifest = String::from("instance,file,scenario,num_jobs,num_machines,seed_hex\n");

    let offset: usize = std::env::var("JSS_INDEX_OFFSET")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for i in offset..(offset + num_instances) {
        let seed = seed_for_index(i as u64, base_seed);
        let ch = Challenge::generate_instance(&seed, scenario)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let filename = format!("instance_{:0width$}.txt", i, width = width);
        let path = out_dir.join(&filename);
        std::fs::write(&path, ch.to_fjsp())?;

        let seed_hex: String = ch.seed.iter().map(|b| format!("{:02x}", b)).collect();
        manifest.push_str(&format!(
            "{},{},{},{},{},{}\n",
            i, filename, scenario, ch.num_jobs, ch.num_machines, seed_hex
        ));
        paths.push(path);
    }

    std::fs::write(out_dir.join("manifest.csv"), manifest)?;
    Ok(paths)
}
