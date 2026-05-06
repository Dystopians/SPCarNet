# Final Stage SCE9 Sentinel-Guided Local Surgery Design

Date: 2026-05-06

Decision: `SCE9_SYNTHETIC_PASS_REAL_COURTYARD_ROLLBACK_ONLY`

## Goal

SCE9 revives local mesh surgery only where evidence demands it. Non-delete edits are triggered by sentinel conflicts and local evidence, not by a global desire to edit topology.

## Operations

- `SNAP_VERTICES`
- `SPLIT_TRIANGLES`
- `FILL_PATCH`
- `APPEARANCE_RESET`
- `PROTECT`
- `REJECT`

## Safety Rules

A proposal is accepted only if it has enough surface support, low free-space risk, and no prior-only hallucination flag. Unknown unobserved voids are rejected by default.

## Current Real-Scene Status

Courtyard ECG evidence currently recommends rollback/protect behavior only. Therefore SCE9 does not promote a real non-delete topology edit for courtyard.

