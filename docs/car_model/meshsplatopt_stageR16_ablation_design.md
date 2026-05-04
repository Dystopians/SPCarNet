# MeshSplatOpt Stage R16 ablation design

Date: 2026-05-03

## Purpose

R16 tracks whether MeshSplatOpt's components are scientifically necessary, not just implementation scaffolding. The current code adds a formal ablation contract so every component has an expected failure mode and evidence status.

## Implemented interface

- `scripts/car_model/meshsplatopt_run_ablation_suite.py`
- contract JSON: `outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_contract.json`
- readable table: `outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_contract.md`
- summary JSON: `outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_summary.json`

## Current status

The interface defines 14 required ablations:

- no CSEF debt;
- no free-space evidence;
- no render gate;
- no sparse geometry gate;
- no changed-pixel gate;
- no rollback;
- no teacher recovery;
- delete/collapse only;
- snap only;
- fill only;
- giant-hole fill without certificate;
- object prior without scene gate;
- budget controller disabled;
- topology freeze disabled.

Current summary:

- total ablations: `14`
- evidence-backed or partially backed: `10`
- interface-ready only: `4`
- gate: `PARTIAL_PASS_NEEDS_PUBLIC_SCENE_COMPLETION`

## Gate status

`PARTIAL_PASS`. R16 now has a complete interface and several negative/partial evidence rows from R17-R28, R43-R56. It is not a full `PASS` until the remaining interface-only ablations are run on synthetic and at least one public scene.

