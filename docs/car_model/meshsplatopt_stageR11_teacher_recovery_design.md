# MeshSplatOpt Stage R11 Teacher Recovery Design

Date: 2026-05-02

## Goal

Define and implement the teacher-guided appearance/geometry recovery contract after accepted edits. R11 must distinguish real renderable recovery from contract-only fallback.

## Recovery Contract

Before edit:

- cache teacher RGB/depth/normal/alpha placeholders when real render outputs are unavailable;
- record visibility and edit-region masks;
- record model/edit metadata.

After edit:

- plan a recovery optimization window;
- preserve unedited regions through teacher distillation terms;
- allow edited regions to match GT/sparse geometry when available;
- initialize fill appearance from neighbors or mark as pending.

## R11 Scope

This stage implements cache/report contracts and CLI. It does not fabricate render metrics. If no renderable model exists, the stage is a documented `SOFT PASS`.

## Gate

`PASS` if real tiny recovery runs. `SOFT PASS` if recovery contract works and missing render path is documented.
