# MeshPrior Stage 5 Design — Optimizer Adapter

| Field | Value |
|---|---|
| Stage | M5 / optimizer adapter |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M4 protect/prune proposals |

## 1. PRISM Presence

PRISM code is present in this repository:

- `train.py` imports and prepares PRISM state,
- `utils/prism_scoring.py` computes `prune_score_t`,
- `utils/prism_counterfactual.py` selects candidates and runs counterfactual gates,
- `utils/prism_pipeline.py` controls multi-round pruning and rollback,
- `utils/prism_validation.py` handles validation and rollback metrics,
- `scripts/parking_ground/*` contains PRISM experiment launchers.

M5 does not patch `train.py`. It exports neutral artifacts that PRISM can consume in a later integration pass.

## 2. Hook Target

The eventual PRISM hook should sit after base PRISM score computation and before candidate selection:

```text
base PRISM scores -> bounded MeshPrior adjustment -> candidate selection -> counterfactual gate
```

The relevant score is `scores.prune_score_t`; protect scores should influence risk/keep/protected behavior rather than directly deleting triangles.

## 3. Neutral Artifact Contract

M5 exports:

```text
meshprior_scores.npz
meshprior_prism_scores.json
```

The NPZ is the generic contract. The JSON is PRISM-oriented but still passive.

## 4. Preventing Prior Override

Default combination is bounded:

```text
keep_score_final = keep_score_base + alpha * clamp(meshprior_protect, 0, 1)
prune_score_final = prune_score_base + beta * clamp(meshprior_prune, 0, 1)
```

Defaults:

```text
alpha <= 0.25
beta <= 0.25
```

MeshPrior may nudge scene scores but cannot dominate them in early experiments.

## 5. Stage Gate

M5 passes if:

- score loading works,
- per-region normalization is finite,
- bounded add respects alpha/beta,
- exported JSON/NPZ can be loaded again,
- adapter reports that PRISM is present.
