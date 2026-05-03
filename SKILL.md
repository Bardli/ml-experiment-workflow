---
name: ml-experiment-workflow
description: Use when starting, iterating, or documenting an ML / medical-imaging experiment in a self-contained folder — covers the L0–L4 information hierarchy (code → CSV → visualizations → PLAN.md → SUMMARY.md), PLAN.md/SUMMARY.md document lifecycle, versioned results folders for buggy vs fixed runs, required per-version visualizations, mandatory wandb logging with a documented metric table, detached background training, watchdogs, docs/INDEX.md cross-referencing, and archival of superseded design docs. Triggers — "set up experiment folder", "new finetune run", "v2 of this experiment", "results changed because of a flag", "write up the experiment", "archive this design doc", "make me a plot to check", "what metrics are we logging".
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
| Repo-level map | — | `docs/INDEX.md` |
| Superseded design | — | `docs/archived/<date>-<name>-superseded.md` |
| User's exact words on a design call | — | `docs/archived/<date>-<topic>-userquote.md` |
| Cross-session state | — | `memory/project_<name>.md` |

## Common Mistakes

- **Running training/eval without wandb.** wandb is mandatory. If `wandb.init` is commented out or stubbed, stop and re-enable it before launching.
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

## Red Flags — STOP and re-read this skill

If you find yourself thinking any of these, you are about to violate the workflow:

- "It's just a sanity check / overfit run / 5 cases — wandb is overkill."
- "I'll add SUMMARY.md / Monitoring table later when there's something to summarise."
- "Let me bump `experiment.name` so the new run goes to a fresh `out_dir` automatically."
- "I'll skip the formula in the Monitoring table — anyone can grep the code."
- "I'll just rename results/ to results_old/ — same idea as `results_v1_<bug>/`."
- "Plots aren't necessary, the CSV is right there."
- "30-minute deadline, I'll cut the doc work and add it after launch."

All of these mean: stop, scaffold the canonical layout, wire wandb, write the metric table, then launch. The skill exists because the same shortcuts have produced the same lost work before.
