# MOF-5: parallel training, random control and shared holdouts

The live AL campaign remains in `outputs/module_B_active_learning_mof5_20bar`.
Its current tmux process is not stopped, restarted or edited by the comparison tools.
Activate `MOF_sim` for every command below.

## Prepared experiment

`outputs/module_B_mof5_al_comparison/comparison.json` freezes the physical model,
0.1–20 bar candidate set, thirteen common initial states, twenty locked holdouts,
24-state training budget, three simulation seeds and one permitted precision retry.
A precision retry uses the existing fourfold simulation-length rule.

The random control uses a deterministic pseudo-random ranking of candidate IDs
(seed 2026), without fitting a GP for selection or reading holdout labels. Both
strategies use the same GP for evaluation. Uniform candidate selection refers to
the fixed candidate pool, which has a deliberately dense extension above 10 bar;
it does not imply a uniform distribution over the continuous pressure interval.

The random control reuses completed common initial states and their retries from
AL. It does not duplicate their simulations. Until all thirteen common initial
states and eligible initial precision retries are finished, `random` reports
`waiting_for_shared_initial_design` and starts no simulation. Five existing
initial states are the frozen baseline. Therefore differences before the common
initial design is complete are not evidence about AL versus random selection.

The holdout worker has a separate campaign, output directory and writer lock.
It prioritizes the new high-pressure holdouts, then the remaining original
holdouts. Existing completed labels are reused. After every holdout has an initial
label, eligible imprecise labels receive at most one precision retry. Labels that
remain imprecise are reported; precision is never guaranteed by a retry.

## Commands

Preparation is idempotent and safe while the AL writer runs. It reads atomic
completed-data snapshots and verifies their physical model and raw-log evidence.
Simulation files for an unfinished request are not imported.

```bash
bash scripts/run_module_b_comparison.sh prepare

# Separate tmux window: up to two new holdout requests, three CPU cores.
bash scripts/run_module_b_comparison.sh holdout --max-requests 2 --jobs 3

# Refresh shared starts and run up to two random requests once the starts are ready.
bash scripts/run_module_b_comparison.sh random --max-requests 2 --jobs 3

# Read-only with respect to both training campaigns; safe while workers run.
bash scripts/run_module_b_comparison.sh report
bash scripts/run_module_b_comparison.sh status
```

Continue AL with its existing `run_module_b_al.sh resume mof5 --root
outputs/module_B_active_learning_mof5_20bar --max-requests 2 --jobs 3` command,
after its current invocation ends. Do not launch a second writer for the same
campaign. Repeat the shared report after each small contingent: it saves immutable
checkpoints of completed training results, not partial runs. Checkpoints from
before comparison preparation cannot be reconstructed if they were never saved.
The random runner also saves a checkpoint after every completed request.

## Assessment decided before the strategy comparison

- Primary metrics: CO2 and N2 RMSE (molecules/cell), plus absolute CO2/N2 selectivity
  MAE. Relative errors and GP interval coverage are supplementary diagnostics.
- Intended comparisons: 13 common initial states, then 16, 20 and 24 training states.
  The report only pairs budgets actually captured for both strategies. It also
  shows error against sums of seed runtimes, including all precision retries.
  This is not elapsed campaign time or a measured CPU-time counter.
- The same available holdout labels evaluate every checkpoint, with a recorded
  test-set hash. Metrics are split at 10 bar and show missing regional labels.
- The existing 20% relative-error scale is an exploratory diagnostic, not an
  application-specific acceptance threshold. Such a threshold still needs a
  physical/use-case justification; it is not chosen after looking at the winner.
- A complete internal assessment requires all twenty holdouts, disclosure of
  imprecise training/test labels and interval coverage, and completed common
  initial designs. Previously inspected holdouts are identified explicitly.
- One random trajectory is a pilot comparison. Replicated random seeds are needed
  before a statistical superiority claim. Another trajectory can be prepared with
  `--random-seed 2027 --root outputs/module_B_mof5_al_comparison_seed2027`; each root
  currently has its own worker, so do not rerun already computed labels blindly.
- Physical validation against reference data remains separate from GP/LAMMPS
  prediction accuracy and from demonstrating an acquisition-policy advantage.

Artifacts: `report/comparison_report.json`, `report/learning_curve.csv`,
`report/learning_curve.png`, `report/simulation_diagnostics.csv`, and `report/report.md`.
The strategy checkpoints can be re-evaluated with a growing shared test set;
all curves in one generated report always use exactly the same labels.

## Existing reference candidates and precision issues

Existing archived reference files are under the live campaign's
`evidence/pilot_evidence/source/references`:

- `10.1039b807470d.Isotherm14.json`: IRMOF-1 CO2/N2, simulation reference, CO2
  fraction 0.1 at 298 K, excess loading in mmol/g, including points near 1, 3, 5,
  10 and 20 bar. The live campaign reports absolute loading at 298.15 K; basis
  conversion and physical-model matching are required before quantitative comparison.
- `10.1016j.seppur.2018.01.013.Isotherm4.json`: experimental metadata at 297 K,
  approximately equimolar CO2/N2; it does not directly match the 298.15 K, 15/85
  initial isotherm. Its zero-pressure point is outside this campaign's domain.
- The DDEC/UFF CO2 and N2 CSV files contain pure-gas reference curves, pressure in
  Pa and loading in mol/kg. They must not be compared directly to mixture labels.

Reference metadata alone does not verify force fields, absolute/excess basis,
framework treatment or experimental sample equivalence. No new reference
simulation is launched automatically.

The 0.1-bar mixed point and the almost-pure-CO2 point near 0.352 bar currently
miss precision after their allowed retry. The latter belongs to the prescribed
initial design, so its selection is not an adaptive-policy failure. Small
minority-component populations and relative errors near zero require explicit
reporting. Inspect the diagnostic CSV before changing tolerances or adding retries.
