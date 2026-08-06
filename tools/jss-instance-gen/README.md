# jss-instance-gen

A small, **self-contained** generator for TIG `job_scheduling` benchmark instances —
a Flexible Job Shop Scheduling Problem (FJSP) with a **total weighted tardiness (TWT)**
objective. You give it a number of instances and a scenario (shop type); it writes that
many instances to disk in the standard FJSP benchmark text format, extended with the
per-job due dates and weights the TWT objective needs.

It has **no dependency on the TIG monorepo** — only two common crates (`rand`,
`rand_distr`). The core generator preserves TIG's RNG sequence, but deliberately floors
generated due dates rather than using TIG's rounding behavior.

---

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (stable, 1.70+). Installing via `rustup`
  gives you the `cargo` build tool used below.
- No network access needed after the first build (it downloads `rand`/`rand_distr` once).

## Build

From the unzipped folder:

```bash
cargo build --release
```

The binary is produced at `target/release/jss-instance-gen`.

## Usage

```bash
jss-instance-gen <NUM_INSTANCES> <SCENARIO> [OUT_DIR] [START_INDEX]
```

| Argument          | Required | Description                                                           |
| ----------------- | -------- | --------------------------------------------------------------------- |
| `<NUM_INSTANCES>` | yes      | How many instances to generate, e.g. `100`.                            |
| `<SCENARIO>`      | yes      | Shop type — one of the five scenarios listed below.                    |
| `[OUT_DIR]`       | no       | Output directory. Defaults to `./instances`.                           |
| `[START_INDEX]`   | no       | First deterministic seed and filename index. Defaults to `0`.          |

### Scenarios (shop types)

| Value              | Description                                          |
| ------------------ | --------------------------------------------------- |
| `flow_shop`        | Flow shop, low machine flexibility.                 |
| `hybrid_flow_shop` | Flow shop with parallel/flexible machines.          |
| `job_shop`         | Classic job shop (varied routes).                   |
| `fjsp_medium`      | Flexible job shop, medium flexibility & reentrance. |
| `fjsp_high`        | Flexible job shop, high flexibility.                |

### Example

```bash
# 100 medium-flexibility FJSP instances into ./benchmark
./target/release/jss-instance-gen 100 fjsp_medium ./benchmark

# A disjoint set using deterministic seed/file indices 150 through 249
./target/release/jss-instance-gen 100 fjsp_medium ./type_b 150
```

You can also run it without building a standalone binary first:

```bash
cargo run --release -- 100 fjsp_medium ./benchmark
```

## Instance sets in this repository

| Set | Location | Seed/file indices | Due-date setting | Speed factors |
| --- | --- | --- | --- | --- |
| Type A | `instances/` (mirrored at `../../benchmarks/math/fjsp_twt/instances/generated/`) | 0–99 | Historical output: `g = 0.5`, rounded | 0.8–1.2 |
| Type B | `instances_type_b/` | 150–249 | `g = 0.75`, floored | 0.8–1.2 |
| Type C | `type C instances/` | 250–349 | `g = 0.75`, floored | 0.2–1.8 |

Indices 100–149 are already used by the separate test set at
`../../benchmarks/math/fjsp_twt/instances/test/`, so Type B begins at 150.

---

## Where instances are saved & how they're named

Everything is written into `OUT_DIR` (default `./instances`):

```
benchmark/
├── instance_000.txt
├── instance_001.txt
├── ...
├── instance_099.txt
└── manifest.csv
```

- **`instance_NNN.txt`** — one file per instance, in standard FJSP format (see below). The
  number is the instance index, zero-padded to at least 3 digits (it widens automatically
  if you ask for ≥ 1000 instances).
- **`manifest.csv`** — a summary table of the whole set, with columns:
  `instance, file, scenario, num_jobs, num_machines, seed_hex`.

### Reproducibility

The set is fully reproducible: instance `i` is generated from a deterministic seed derived
from its index, so running the same command again produces identical files. `START_INDEX`
allows multiple sets to use disjoint index/seed ranges. The exact seed used for each instance
is recorded in `manifest.csv` (`seed_hex`).

> All TIG `job_scheduling` instances currently have **50 jobs**; the number of machines is
> determined by the scenario (30 for most scenarios).

---

## Output format (extended FJSP)

Each `instance_NNN.txt` is the classic Brandimarte / Hurink / Fattahi FJSP layout used by
most published FJSP datasets and solvers, **plus two trailing lines** carrying the
total-weighted-tardiness data:

```
<num_jobs> <num_machines>
<line for job 1>
<line for job 2>
...
<line for job n>
DUE_DATES: d_0 d_1 ... d_{n-1}
WEIGHTS: w_0 w_1 ... w_{n-1}
```

Each **job line** is:

```
<num_operations>  <k> <m> <p> <m> <p> ...   <k> <m> <p> ...
```

- `<num_operations>` — number of operations in the job.
- Then, for **each operation** in sequence:
  - `<k>` — the number of machines that can run this operation,
  - followed by `<k>` pairs of `<machine_id> <processing_time>`.

The two trailing lines give, in job order:

- `DUE_DATES:` — the due date `d_j` of each job. A job finishing after `d_j` is tardy.
- `WEIGHTS:` — the tardiness weight `w_j` of each job (an integer in `{1, 2, 3}`).

The objective minimised by the challenge is **total weighted tardiness**:
`TWT = Σ_j w_j · max(C_j − d_j, 0)`, where `C_j` is job `j`'s completion time.

Notes:

- **Machine ids are 1-indexed** (`1 .. num_machines`), matching the convention in published
  FJSP benchmark files.
- Operations are listed in their precedence order (operation _o_ must finish before
  operation _o+1_ of the same job starts).
- A job's line index (0-based, in file order) equals its job index in `DUE_DATES`,
  `WEIGHTS`, and in TIG itself.
- The two `DUE_DATES:`/`WEIGHTS:` lines are appended **after** the `num_jobs` job lines, so
  a plain FJSP/makespan parser that reads only the header and job lines still works and
  simply ignores them.

### Worked snippet

```
50 30
3  2 1 80 28 97  3 2 38 10 39 24 47  1 3 6
...
DUE_DATES: 3766 3766 3766 3863 ...
WEIGHTS: 3 1 2 1 ...
```

Reading the first job line `3  2 1 80 28 97  3 2 38 10 39 24 47  1 3 6`:

- `3` → the job has 3 operations.
- Operation 1: `2 1 80 28 97` → 2 eligible machines: machine `1` takes `80`, machine `28`
  takes `97`.
- Operation 2: `3 2 38 10 39 24 47` → 3 eligible machines: `2`→`38`, `10`→`39`, `24`→`47`.
- Operation 3: `1 3 6` → 1 eligible machine: machine `3` takes `6`.

Then `DUE_DATES`/`WEIGHTS` give this job (index 0) a due date of `3766` and a weight of `3`.

## Using the library directly

The generator is also a library if you'd rather call it from your own Rust code:

```rust
use jss_instance_gen::{generate_instances, Challenge, Scenario};
use std::path::Path;

// Write a whole set to disk:
generate_instances(100, Scenario::FJSP_MEDIUM, Path::new("./benchmark"), None)?;

// Or generate a single instance and get its (extended) FJSP text:
let ch = Challenge::generate_instance(&[0u8; 32], Scenario::FJSP_MEDIUM)?;
let fjsp_text: String = ch.to_fjsp();
// ch.due_dates and ch.weights are also available directly.
```
