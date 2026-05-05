---
name: ml-experiment-workflow
description: Use when starting, iterating, or documenting an ML / medical-imaging experiment in a self-contained folder — covers the L0–L4 information hierarchy (code → CSV → visualizations → PLAN.md → SUMMARY.md), PLAN.md/SUMMARY.md document lifecycle, versioned results folders for buggy vs fixed runs, required per-version visualizations, mandatory wandb logging with a documented metric table, mandatory per-file function manifests + repo-level docs/CODE_INDEX.md for cross-experiment code reuse, detached background training, watchdogs, docs/INDEX.md cross-referencing, archival of superseded design docs, and a mandatory repo-root CLAUDE.md (skills list + folder tree + env + TODO) so every Claude session orients without grepping. Triggers — "set up experiment folder", "new finetune run", "v2 of this experiment", "results changed because of a flag", "write up the experiment", "archive this design doc", "make me a plot to check", "what metrics are we logging", "is there already a function that does X", "add a helper for Y", "refactor this into utils".
---

# ML Experiment Workflow

## Overview

A single experiment is one folder. That folder holds the **plan, code, config, data symlinks, training logs, and every version of results** — past and present. Outside the folder we only touch the source tree when the model itself needs a code change. Documents (`PLAN.md` → `SUMMARY.md`) describe state; folder names (`results_v2_buggyinfer/`) preserve history.

**Core rule:** never overwrite results when a bug is found upstream. Rename the old folder with the bug name, and create a new `results_vN/` so we can compare.

## Information Hierarchy (the tower)

Every experiment maintains five layers. Higher layers consume lower ones; the user reads top-down (L4 → L3 → L2), Claude produces bottom-up (L0 → L1 → L2 → keeps L3 in sync → writes L4 at the end). **The user does not read L0/L1.** L2 is the verification layer — without it, the user has no way to check a run.

| Level | Artifact | Producer | Consumer | Per-version? |
|---|---|---|---|---|
| **L0** | code, `config.yaml`, `run.sh`, training/inference scripts | Claude | the machine | shared (config diff is the version) |
| **L1** | result CSVs, checkpoints, `train.log`, tensorboard, wandb | the run | L2/L3 generators | **yes** — `results_vN_<tag>/`, `exp_log_vN_<tag>/` |
| **L2** | **visualizations** — DSC curves, before/after overlays, per-case mosaics, holdout comparisons | Claude (after every version) | the user (verification) | **yes** — `results_vN_<tag>/viz/` |
| **L3** | `PLAN.md` — design + implementation + chronological §N for each version | Claude (kept live) | the user | one file, append §N per version |
| **L4** | `SUMMARY.md` — one-screen consolidation pointing back into L3/L2/L1 | Claude (when done) | the user, downstream readers, `docs/INDEX.md` | one file, rewritten on `vN_final` |

**Required deliverable rule:** a version is not "done" until its `results/viz/` (or `results_v<N>_<tag>/viz/` after bump) contains at least one plot the user can scan. CSVs alone are not a deliverable — they are L1. If you're tempted to report a result without producing the corresponding L2, stop and generate the plot first.

*At scaffold-time (before the first run), the `viz/` folder should exist but will be empty — that's fine. The "done" gate fires when reporting results, not when laying out folders.*

## When to Use

- Starting any new training/finetune/eval experiment that will produce checkpoints + metrics.
- Re-running with a changed flag, dataset version, or fixed bug — i.e. need a new `vN`.
- Writing the final summary, or publishing the experiment to `docs/INDEX.md`.
- User asks to "archive" or "supersede" an existing design doc.

Skip when: a one-off script with no checkpoints and no comparison ever needed.

## Iteration Protocol — Smoke Test Before Full Run

Two phases. Skipping phase 1 is the most common way to waste a day of cluster time and tank the lab's LevelFS.

### Phase 1 — Interactive smoke test (20–40 GB MIG, 5–10 samples)

1. Request a **20 GB or 40 GB H100 MIG** interactive allocation — not a full 80 GB. (Cluster sizing rules live in the `fir-cluster` skill.)
2. Copy or symlink **5–10 training samples** into `data/`. The whole point is fast iteration — keep it tiny.
3. Build the full training + eval pipeline end-to-end: data loader → model → loss → optimizer step → wandb log. Wire wandb on the **very first** run (this skill's mandatory rule applies even at 5 cases).
4. Run for enough steps that loss visibly decreases. Estimate per-epoch wall-clock for the full dataset — that estimate is your phase-2 sizing input.

### Phase 1 — Live resource verification (REQUIRED, in a second terminal)

Wandb dashboards lag by ~30 s and don't show CPU/RAM. While the smoke run is going, open another terminal and ssh directly into the compute node:

```bash
sq                              # find the node your job is on
ssh <node>                      # ssh straight into it (Fir allows this)
nvidia-smi                      # GPU memory + utilization
htop                            # CPU
free -h                         # RAM
```

The bar to clear, **all three**:

- **GPU memory ≥ ~95%** of allocated VRAM. If only 30% used → you over-allocated, drop to a smaller MIG slice.
- **GPU utilization ≥ ~90%** sustained. If 30–60% → dataloader / I/O bottleneck. Increase `num_workers`, prefetch, or move data closer to compute. **Do not scale to the full job until this is fixed** — the bottleneck multiplies on a bigger GPU.
- **CPU saturation** matches the breakeven count for the GPU slice (1/3/5/12 for 1g/2g/3g/full H100).

### Phase 1 — Train-set sanity inference (proves no train/infer skew)

Run the same eval/inference script you'll use in phase 2, but on the **5–10 training samples**. Expected metrics: **~100% DSC / accuracy / IoU.** Anything materially lower means a train/inference flag mismatch — exactly the `keyframe_bypass_bug` class of issue this skill's versioning rule exists to recover from. Cheaper to catch in phase 1 than after a 24-hour full job.

### Phase 2 — One-day full-dataset job (only after phase 1 clears all three bars)

- Submit a SLURM batch job on a **full H100-80g** with `--time=1-00:00:00` (1-day jobs go on full H100, per the fir-cluster skill).
- Save **latest** and **best** checkpoints every epoch (or every N steps for long epochs). Both, every time — `latest` for resume, `best` for eval.
- Implement **patience-based early stopping**: halt if best validation metric hasn't improved for ~10 epochs/steps. Don't burn shared LevelFS on a flat curve.
- Watch wandb validation curves trend upward. If they don't within the first ~10–20% of total steps, kill the job and diagnose — don't let a bad run consume the whole 24 h.

> **The non-negotiable phase-1 exit gate:** wandb shows decreasing training loss + GPU memory & utilization both ~100% (verified by `nvidia-smi` on-node, not just wandb) + train-set inference metrics ~100%. **All three.** Until then you are not ready for the full dataset.

## Folder Layout (canonical, mirrors `finetune10_eay131/`)

```
<experiment_name>/
├── PLAN.md                       living plan, status header at top, §-numbered
├── SUMMARY.md                    written when experiment is "done"
├── config.yaml                   single source of truth for hyperparams
├── run.sh                        detached training launcher (nohup setsid …)
├── stage_data.py                 idempotent symlink builder for data/
├── infer_*.py / eval_*.py        inference + per-epoch eval drivers
├── data/                         symlinks ONLY — never copies
├── data_holdout/                 zero-overlap holdout symlinks (if any)
├── exp_log/                      L1 — current-version training output (ckpts, tb, wandb)
├── exp_log_vN_<bugname>/         L1 — renamed when a bug invalidates the run
├── results/                      L1 — current-version eval CSVs
│   └── viz/                      L2 — required plots/overlays for THIS version
├── results_vN_<bugname>/         L1 — preserved buggy results (do not delete)
│   └── viz/                      L2 — plots that show *why* this version was bad
└── results_vN_with_<fix>/        L1 — re-eval of same ckpts under fixed conditions
    └── viz/                      L2 — plots demonstrating the fix's effect
```

`stage_data.py` must be idempotent (re-runs are no-ops). Symlink, never copy.

## Monitoring (mandatory wandb)

Every training and evaluation run **must** log to wandb. No exceptions. Local CSVs and tensorboard are L1 artifacts; wandb is the cross-version, queryable record we read against.

**The size of the experiment is not an exemption.** Overfit sanity checks log to wandb. 5-case smoke tests log to wandb. Single-epoch dry runs log to wandb. The reason is not "we need fancy monitoring" — it's that wandb is the only artifact that survives `rm -rf` of the workspace and lets v3 of an experiment compare against v1. JSONL + `tail -f` does not give you that.

Forbidden rationalizations (each has been used before, do not repeat):

| Excuse | Reality |
|---|---|
| "Just a 5-case overfit, JSONL is faster" | JSONL is local, ephemeral, and not cross-comparable. wandb takes 4 lines to set up. Add them. |
| "I'll add wandb in v2 if it works" | v2 needs to compare against v1's curves, which won't exist. Wire it now. |
| "wandb is offline / no key — let me skip it" | Run `wandb login` or set `WANDB_API_KEY`. If genuinely offline, use `mode='offline'` and sync later — never `mode='disabled'`. |
| "Tensorboard is enough" | Tensorboard files live in `exp_log/`, which gets renamed when a bug lands. wandb's run history persists. |

**Run conventions:**
- `project` = experiment folder name (e.g. `finetune10_eay131`).
- `name` = `vN_<tag>` matching the corresponding `results_vN_<tag>/` folder.
- `config` = full `config.yaml` dumped at `wandb.init` time (`wandb.config.update(yaml.safe_load(...))`).
- `tags` = `["train" or "eval", "vN", "<dataset>", "<model_variant>"]`.
- Capture `wandb.run.get_url()` at startup and write it into the **PLAN.md status header** so the user can click through.

**Required PLAN.md section — `## N. Monitoring`** must list every logged metric in a table. The user reads this to confirm *what we are measuring* and *that the formula matches the published convention*. If a metric isn't in the table, don't log it; if it's logged but not in the table, the doc is wrong.

| W&B key | What it measures | How it's calculated | Source |
|---|---|---|---|
| `train/loss_total` | combined training loss per step | weighted sum, weights from `config.loss` (e.g. `α·focal + β·dice + γ·iou`) | `training/loss_fns.py:compute()` |
| `train/loss_dice` | dice term only | `1 − 2·∑(p·y) / (∑p + ∑y + ε)`, ε=1e-6, per-batch mean | `training/loss_fns.py:dice_loss()` |
| `val/dsc_keyframe` | DSC on the prompted keyframe slice | `2·|p∩y| / (|p|+|y|)`, prediction binarised at 0.5 | `eval_sweep.py:eval_keyframe()` |
| `val/dsc_propagation` | DSC on non-prompted slices | same formula, mean over non-keyframe slices in same volume | `eval_sweep.py:eval_propagation()` |
| `val/dsc_mean` | per-volume mean DSC over all annotated slices | `mean(per_slice_dsc)` per volume, then mean over volumes | `eval_sweep.py:eval_volume()` |
| `lr` | current learning rate | from optimizer scheduler | `training/trainer.py` |
| `system/gpu_mem` | peak GPU memory used | `torch.cuda.max_memory_allocated()` (auto-logged) | wandb |

The table above is a **template** — replace rows with your actual metrics. Required columns: **W&B key, What it measures, How it's calculated, Source**. The "How it's calculated" cell must contain the actual formula (or one-line algorithmic description), not just "DSC" or "see code".

## PLAN.md → SUMMARY.md Lifecycle

**PLAN.md** is the living document, written **before code**. Structure:

1. **Status header** at the very top — one paragraph, updated each iteration. Include version, date, key metric, **wandb run URL**, and `→ see §N` pointing at the latest results section.
2. **Numbered sections** (`## 1. Goal`, `## 2. Folder layout`, `## 3. Data`, `## 4. Code change`, `## 5. Training`, `## 6. Inference`, `## 7. Monitoring` *(the wandb metric table — required)*, `## 8. Protocol`, …).
3. **Append, don't rewrite**: when a new version finishes, add `## 12. v2 results & root cause` and `## 13. v3 final results` rather than mutating §5/§6 in place. The plan becomes a chronological narrative; the status header always points readers to the freshest §.
4. **Root-cause sections**: when a bug is found, write a `## N. Root cause` section that names the flag/file/line and the symptom (e.g. "keyframe DSC stuck at 0.04"), and update the *training* and *inference* sub-sections side-by-side — both must match.

**SUMMARY.md** is the consolidation, written when the experiment hits its target (or is abandoned). Structure: data → finetune → inference → results table → caveats. Numbers only, no narrative of how we got there. Keep it under one screen.

## Versioning Rule (the one that stops bugs from hiding)

**Naming is fixed, not negotiable.** When you bump versions, the new folder name is **exactly** `results_v<N>_<tag>/` and `exp_log_v<N>_<tag>/`, where `<tag>` is a short bug-or-fix label (`keyframe_bypass_bug`, `with_inference_override`, `final`). Forbidden alternatives:

- ❌ Bumping `experiment.name` in config and letting `out_dir` rotate.
- ❌ `run_2/`, `attempt_3/`, `new_results/`, `latest/`.
- ❌ Timestamp-only directories (`2026_05_03_1730/`) — they don't tell you *what* changed.
- ❌ Overwriting `results/` and trusting git to recover the old one.

The current run lives at `results/` (no version suffix) until it's bumped or finalized. On bump: `mv results results_v<N>_<tag>/` first, *then* re-run.

When a bug invalidates a run:

1. `mv exp_log exp_log_vN_<bugname>` and `mv results results_vN_<bugname>` (with its `viz/` intact — the bad plots are evidence).
2. Fix the code/config. **Audit both training-side and inference-side** — flags like `use_mask_input_as_output_without_sam` must match on both sides.
3. Re-run, producing new `exp_log/` and `results/`.
4. Optional: re-evaluate the *old* checkpoints under the *new* inference and save as `results_vN_with_<fix>/` to prove the bug was inference-side, not training-side (or vice versa).
5. **Generate L2 visualizations into `results/viz/`** — at minimum a metric curve and a before/after overlay vs the prior version.
6. Add a `§N` to PLAN.md naming the bug, the fix, the delta, and **embedding/linking the L2 plot** so the reader can verify without opening CSVs.

Never `rm -rf` a buggy results folder. The diff between buggy and fixed is the experiment's most useful artifact.

## Code Reuse & Code Index

Cross-experiment code reuse is mandatory. Two-part rule: every file declares what's in it, and a repo-level index makes those declarations searchable. The pattern this skill exists to prevent: experiment N+1 silently re-implements (often subtly differently) a function that already exists in experiment N — and the divergence is only discovered when the two experiments' results disagree.

### Per-file manifest (top of every Python file)

Every `.py` file in an experiment (training, eval, inference, viz, utils, stage_data, etc.) opens with a module docstring that lists the **public functions** in the file with purpose + full typed signature. This is what future-you (or future-Claude) reads to decide "do I already have this?" without opening the body.

Required format:

```python
"""eval_sweep.py — per-epoch DSC sweep over a holdout set.

Functions:
    eval_keyframe(pred_dir: Path, gt_dir: Path) -> dict[str, float]
        DSC on the prompted keyframe slice for every case.
        In:  prediction NIfTI dir, ground-truth NIfTI dir.
        Out: {case_id: dsc}.
    eval_propagation(pred_dir: Path, gt_dir: Path) -> dict[str, float]
        Same, but excludes the keyframe (measures SAM2 propagation only).
    eval_volume(pred_dir: Path, gt_dir: Path) -> dict[str, float]
        Per-volume mean DSC across all annotated slices.
"""
```

Rules:
- List **public** functions only (skip `_underscore_helpers`).
- Each entry: one-line purpose + signature with types, plus explicit **In:** / **Out:** lines when the types alone aren't self-explanatory (dict shapes, tensor layouts, file conventions).
- Manifest order matches definition order in the file.
- Signature changes update the manifest in the **same commit**. A drifted manifest is worse than no manifest.
- If a private helper grows past ~30 lines and looks generally useful, promote it (rename, document, add to manifest).

### Repo-level `docs/CODE_INDEX.md`

One markdown file at the repo root maps `function → file → one-line purpose`, grouped by topic (data, training, eval, viz, infra). It's the cross-experiment search surface — the per-file manifest is the in-experiment one.

Format:

```markdown
## Eval

| Function | File | Purpose |
|---|---|---|
| `eval_keyframe` | `finetune10_eay131/eval_sweep.py` | DSC on prompted keyframe slice |
| `eval_propagation` | `finetune10_eay131/eval_sweep.py` | DSC on non-keyframe slices |
| `compute_per_volume_dsc` | `finetune218_eay131_split80_20/eval_volume.py` | Per-volume mean DSC |

## Data staging

| Function | File | Purpose |
|---|---|---|
| `symlink_split` | `finetune10_eay131/stage_data.py` | Idempotent train/val symlink builder |
```

### The reuse loop (mandatory, before writing any non-trivial function)

1. **Grep `docs/CODE_INDEX.md` first** for the concept (`dsc`, `load_npz`, `stage_data`, `overlay`, …). If you find a match, **import it** — do not re-implement.
2. If no exact match but a close one exists, **read its file's manifest docstring**. Extend the existing function via a parameter rather than forking it into a new experiment.
3. Only when nothing fits, write a new function — and **update the file's manifest + `docs/CODE_INDEX.md` in the same commit** that introduces it.
4. If you copy-paste a function from another experiment instead of importing it, that is a forking event — extract it into a shared module (`shared/`, `common/`, or a sibling `utils.py` imported by both) before the second copy lands.

Forbidden rationalizations (each has been used before, do not repeat):

| Excuse | Reality |
|---|---|
| "It's a 5-line helper, not worth indexing." | Five-line helpers proliferate fastest. The index is what stops the same DSC formula from being implemented six different ways. |
| "I'll add it to the index after the run lands." | After the run, no one updates the index. Add the row in the **same commit** as the function. |
| "Manifest docstrings are noise — the function is right there." | The manifest is read across experiments by people (and Claude sessions) who don't know the function exists yet. Visible signature beats `grep` archaeology. |
| "I'll just copy-paste from `finetune10` into the new experiment." | Copy-paste forks the bug surface. Import from the original; if it has to change, change the original or extract a shared module — never run two copies of the same logic. |
| "Types/In/Out lines are obvious from the signature." | They are obvious to *you, today*. The manifest is for the reader six months out who has to decide whether to reuse or rewrite. Spell it out. |

## Background Training & Watchdogs

Long jobs MUST survive ssh/laptop close. Pattern:

```bash
# inside run.sh
nohup setsid python -u training/train.py \
    --config-path <abs> --config-name config.yaml \
    > exp_log/train.log 2>&1 < /dev/null &
disown
echo $! > exp_log/train.pid
```

Pair with a watchdog (background `Bash` tool call) that tails the log until the process exits and reports completion + final epoch + best metric. Do NOT poll in a sleep loop from the main thread — use `run_in_background`.

## Repo-level `CLAUDE.md` (mandatory — auto-loaded orientation)

`CLAUDE.md` at the repo root is auto-loaded into every Claude Code session in this project. It is the **first thing Claude reads**, before any `grep` / `ls` / `find`. Without it, every new session burns tokens (and your time) re-discovering the layout. With it, Claude jumps straight to the right experiment folder, the right venv, the right cluster account, and the right next task.

**Required sections** (keep the whole file under ~200 lines so it stays in context):

```markdown
# <repo name>

One-paragraph project purpose.

## Skills to invoke (auto-load)

- `ml-experiment-workflow` — for any new experiment / vN bump / SUMMARY write-up
- `ccdb-clusters` — for sbatch / module / account / seff work on Fir
- `medical-imaging-pipeline` — DICOM ↔ NIfTI conversion, RTSTRUCT rasterisation
- `nnunet-converter` — when prepping data for nnUNet
- `<other repo-specific skills>`

## Folder structure (read this instead of `find .`)

```
<repo>/
├── docs/
│   ├── INDEX.md            map of all experiments
│   ├── CODE_INDEX.md       cross-experiment function index — grep BEFORE writing
│   └── DEVLOG.md           running dev log
├── experiments/
│   ├── full_train/runs/baseline/        production training run
│   ├── full_train/runs/baseline-smoke/  phase-1 smoke test
│   └── <expN>/PLAN.md                   per-experiment plan
├── training/                 model + trainer + loss
├── sam2/                     model architecture
├── scripts/ours/             our launchers / one-off helpers
├── data/                     symlinks only (read-only shared dataset)
└── .venv/                    project venv (python 3.11.5, triton 3.6.0)
```

(Annotate every top-level dir in one line. If a dir is large, link its own README.)

## Environment

- Cluster: Fir (`$CC_CLUSTER=fir`); use `def-jma-ab_gpu` GPU account by default — verify with `~/.claude/skills/ccdb-clusters/scripts/show-fairshare.sh`
- Venv: `source /scratch/baidu/<repo>/.venv/bin/activate` (or whatever this repo uses)
- Module loads needed before `claude`: `module load nodejs/20.16.0 python/3.11.5`
- Storage: never write to `$HOME` — use `$SCRATCH`

## Conventions

- All training submitted via `sbatch`, never on the login node.
- Wandb is mandatory on every run, including 5-case smoke tests (see `ml-experiment-workflow`).
- Versioned results: never overwrite `results/` — rename to `results_vN_<bug>/`.

## TODO

- [ ] <next concrete step the user wants done>
- [ ] <…>
- [x] <recently completed, kept here for handoff context until rolled into PLAN.md / DEVLOG.md>
```

**Maintenance rules:**

1. **Update `## TODO` at the end of every working session.** A future Claude (or a Monday-morning-you) reads this list to pick up the work without replaying the chat. Move completed items to `[x]` for one session, then prune them into `docs/DEVLOG.md` or the relevant `PLAN.md`.
2. **Keep the folder tree current.** When you add `experiments/<newexp>/`, add a one-line entry in `CLAUDE.md`. Same commit. (A wrong tree is worse than no tree — Claude will trust it and not re-grep.)
3. **Skills list is curated, not exhaustive.** Only list skills actually used in this repo. Don't paste the full installed-skills catalogue.
4. **No secrets.** `CLAUDE.md` is checked in. Personal account names / paths that shouldn't be public belong in `~/.claude/projects/.../memory/personal_*.md`, not here.

If a session opens and `CLAUDE.md` is missing, stale, or doesn't list the skills → the first action is to repair it, not to start grepping.

## docs/INDEX.md and Archival

The repo's `docs/INDEX.md` is the map. When an experiment finishes:

1. Add one row to the relevant table in `docs/INDEX.md` linking to `SUMMARY.md` (or copy the summary into `docs/<date>-<experiment>-summary.md` if cross-project).
2. **Archive superseded design docs**: move to `docs/archived/<date>-<name>-superseded.md`. Never delete — they explain *why* the current design exists.
3. **Verbatim before paraphrase**: if the user resolves an open design question inline in chat, save their exact wording to `docs/archived/<date>-<topic>-userquote.md` *before* paraphrasing into the live doc.

## Memory Hygiene (cross-session continuity)

Each experiment gets one `project_<name>.md` memory file (see `~/.claude/projects/.../memory/`). Update it — don't accumulate stale duplicates. Schema:

```
Status: <vN complete YYYY-MM-DD>; key metric = <value>; next step = <…>
Why: <reason this experiment exists>
How to apply: <what future-Claude should do when it sees related work>
```

When a root-cause fix supersedes earlier "bugs," mark the old memories `SUPERSEDED by <fix>` rather than deleting — same logic as `docs/archived/`.

## Quick Reference

| Action | Layer | Where it lives |
|---|---|---|
| Hyperparams (single source) | L0 | `<exp>/config.yaml` |
| Detached launch | L0 | `<exp>/run.sh` (nohup setsid … & disown) |
| Current run CSVs / ckpts / logs | L1 | `<exp>/exp_log/`, `<exp>/results/` |
| Buggy run CSVs / ckpts / logs (preserved) | L1 | `<exp>/exp_log_vN_<bug>/`, `<exp>/results_vN_<bug>/` |
| **wandb run (mandatory)** | L1 | `wandb` project = `<exp>`, run = `vN_<tag>`; URL pinned in PLAN.md status header |
| Metric definitions & formulas | L3 | `<exp>/PLAN.md` § Monitoring (one row per logged W&B key) |
| **Visualizations for verification (required per version)** | **L2** | `<exp>/results*/viz/` (curves, overlays, mosaics) |
| Plan, status header, §-numbered history (links L2 plots) | L3 | `<exp>/PLAN.md` |
| Final one-screen write-up (links L2 plots) | L4 | `<exp>/SUMMARY.md` |
| **Auto-loaded session orientation (mandatory)** | — | `CLAUDE.md` at repo root — skills list, folder tree, env, TODO |
| Repo-level map | — | `docs/INDEX.md` |
| **Per-file function manifest (mandatory)** | L0 | top-of-file docstring in every `.py` — public functions with typed signatures + In/Out |
| **Cross-experiment code index (mandatory)** | — | `docs/CODE_INDEX.md` — grep here BEFORE writing a new function |
| Superseded design | — | `docs/archived/<date>-<name>-superseded.md` |
| User's exact words on a design call | — | `docs/archived/<date>-<topic>-userquote.md` |
| Cross-session state | — | `memory/project_<name>.md` |

## Common Mistakes

- **Running training/eval without wandb.** wandb is mandatory. If `wandb.init` is commented out or stubbed, stop and re-enable it before launching.
- **Skipping phase 1 sanity inference and discovering a train/infer flag mismatch only on full-dataset eval.** The 5–10-sample train-set inference is the cheapest way to catch this — if it returns <99% you have a bug, period.
- **Submitting the full-dataset job before GPU utilization is verified ~100% on-node.** Wandb shows allocated memory, not actual utilization — `nvidia-smi` on the running node is the only ground truth.
- **Skipping early-stopping and letting a flat-curve job run 24 h.** Patience early-stop is shared-LevelFS hygiene, not just a convenience.
- **Logging metrics that aren't in the PLAN.md `## Monitoring` table** (or vice versa). The table and the wandb stream must agree, key-for-key, with formulas spelled out — no "see code" placeholders.
- **Reporting a result without an L2 plot.** CSVs are L1 — the user does not read them. Generate the visualization first, then report.
- **Overwriting `results/` when a bug is found.** The diff (and the *plot* of the diff) is the experiment. Rename, don't overwrite.
- **Deleting `viz/` when archiving a buggy version.** The bad plot is evidence — keep it next to its CSVs.
- **Fixing only one side of a flag.** Train-time and inference-time configs must be audited together. (e.g. `use_mask_input_as_output_without_sam` must match.)
- **Mutating §5 when a v2 lands.** Append `§12 v2 results` instead — the plan is chronological.
- **Skipping the L2 → L3 link.** Each PLAN.md §N for a version must reference its `viz/` plots by relative path so the user can scan without leaving the doc.
- **Polling a long job in the main thread.** Use `run_in_background` and a watchdog.
- **Paraphrasing the user before archiving their words.** Verbatim → `docs/archived/`, *then* paraphrase.
- **Letting `MEMORY.md` collect stale duplicates.** Update the existing entry; mark superseded ones explicitly.
- **Missing or stale `CLAUDE.md`.** Without it, every new session re-greps the tree to find the venv, the experiments folder, the cluster account. Repair it before starting any other task. The TODO list at the bottom is the handoff between sessions — keep it current.
- **Re-implementing a function that already exists in another experiment** because nobody grepped `docs/CODE_INDEX.md`. The reuse loop is the cheapest step in this workflow — skipping it produces silent divergence that only surfaces when two experiments' numbers disagree.
- **Adding a function without a top-of-file manifest entry, or without a row in `docs/CODE_INDEX.md`.** Undocumented functions are invisible to the next session — they will be re-implemented.
- **Copy-pasting a helper from one experiment into another.** That is a forking event. Import from the original or extract to a shared module before the second copy lands.
- **Letting a manifest drift from the actual signatures.** A wrong manifest is worse than none — it sends readers down the wrong path. Update it in the same commit as the signature change.

## Red Flags — STOP and re-read this skill

If you find yourself thinking any of these, you are about to violate the workflow:

- "It's just a sanity check / overfit run / 5 cases — wandb is overkill."
- "I'll add SUMMARY.md / Monitoring table later when there's something to summarise."
- "Let me bump `experiment.name` so the new run goes to a fresh `out_dir` automatically."
- "I'll skip the formula in the Monitoring table — anyone can grep the code."
- "I'll just rename results/ to results_old/ — same idea as `results_v1_<bug>/`."
- "Plots aren't necessary, the CSV is right there."
- "30-minute deadline, I'll cut the doc work and add it after launch."
- "I'll just submit the full-dataset job — interactive smoke is wasted time."
- "GPU memory looks full in wandb, that's enough — no need to ssh in and check `nvidia-smi`."
- "Train-set inference at ~80% is good enough; we'll get 100% after more training."
- "Quick helper, no need to add a manifest line or update `CODE_INDEX.md`."
- "I'll just copy this function over from the other experiment — faster than importing."
- "I'll grep for it later if I need it again."
- "Manifest is obvious from the signature, skipping it."

All of these mean: stop, scaffold the canonical layout, wire wandb, write the metric table, **grep `docs/CODE_INDEX.md` before writing a new function**, run phase 1 with all three exit gates verified, then launch. The skill exists because the same shortcuts have produced the same lost work before.
