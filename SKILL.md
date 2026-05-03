---
name: ml-experiment-workflow
description: Use when starting, iterating, or documenting an ML / medical-imaging experiment in a self-contained folder — covers the L0–L4 information hierarchy (code → CSV → visualizations → PLAN.md → SUMMARY.md), PLAN.md/SUMMARY.md document lifecycle, versioned results folders for buggy vs fixed runs, required per-version visualizations, detached background training, watchdogs, docs/INDEX.md cross-referencing, and archival of superseded design docs. Triggers — "set up experiment folder", "new finetune run", "v2 of this experiment", "results changed because of a flag", "write up the experiment", "archive this design doc", "make me a plot to check".
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

**Required deliverable rule:** a version is not "done" until its `results_vN_<tag>/viz/` exists with at least one plot the user can scan. CSVs alone are not a deliverable — they are L1. If you're tempted to report a result without producing the corresponding L2, stop and generate the plot first.

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

## PLAN.md → SUMMARY.md Lifecycle

**PLAN.md** is the living document, written **before code**. Structure:

1. **Status header** at the very top — one paragraph, updated each iteration. Include version, date, key metric, and `→ see §N` pointing at the latest results section.
2. **Numbered sections** (`## 1. Goal`, `## 2. Folder layout`, `## 3. Data`, `## 4. Code change`, `## 5. Training`, `## 6. Inference`, `## 7. Protocol`, …).
3. **Append, don't rewrite**: when a new version finishes, add `## 12. v2 results & root cause` and `## 13. v3 final results` rather than mutating §5/§6 in place. The plan becomes a chronological narrative; the status header always points readers to the freshest §.
4. **Root-cause sections**: when a bug is found, write a `## N. Root cause` section that names the flag/file/line and the symptom (e.g. "keyframe DSC stuck at 0.04"), and update the *training* and *inference* sub-sections side-by-side — both must match.

**SUMMARY.md** is the consolidation, written when the experiment hits its target (or is abandoned). Structure: data → finetune → inference → results table → caveats. Numbers only, no narrative of how we got there. Keep it under one screen.

## Versioning Rule (the one that stops bugs from hiding)

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
| **Visualizations for verification (required per version)** | **L2** | `<exp>/results*/viz/` (curves, overlays, mosaics) |
| Plan, status header, §-numbered history (links L2 plots) | L3 | `<exp>/PLAN.md` |
| Final one-screen write-up (links L2 plots) | L4 | `<exp>/SUMMARY.md` |
| Repo-level map | — | `docs/INDEX.md` |
| Superseded design | — | `docs/archived/<date>-<name>-superseded.md` |
| User's exact words on a design call | — | `docs/archived/<date>-<topic>-userquote.md` |
| Cross-session state | — | `memory/project_<name>.md` |

## Common Mistakes

- **Reporting a result without an L2 plot.** CSVs are L1 — the user does not read them. Generate the visualization first, then report.
- **Overwriting `results/` when a bug is found.** The diff (and the *plot* of the diff) is the experiment. Rename, don't overwrite.
- **Deleting `viz/` when archiving a buggy version.** The bad plot is evidence — keep it next to its CSVs.
- **Fixing only one side of a flag.** Train-time and inference-time configs must be audited together. (e.g. `use_mask_input_as_output_without_sam` must match.)
- **Mutating §5 when a v2 lands.** Append `§12 v2 results` instead — the plan is chronological.
- **Skipping the L2 → L3 link.** Each PLAN.md §N for a version must reference its `viz/` plots by relative path so the user can scan without leaving the doc.
- **Polling a long job in the main thread.** Use `run_in_background` and a watchdog.
- **Paraphrasing the user before archiving their words.** Verbatim → `docs/archived/`, *then* paraphrase.
- **Letting `MEMORY.md` collect stale duplicates.** Update the existing entry; mark superseded ones explicitly.
