# PRISM Reviewer-Risk Checklist

Date: 2026-05-02

## Claim Risks

| risk | status | mitigation |
|---|---|---|
| Claiming universal image-quality dominance | active risk | State that `courtyard` LPIPS trades off; use topology-control framing. |
| Claiming radar-only or scan-only reconstruction | active risk | State assumption: calibrated images plus COLMAP-style sparse geometry. |
| Over-centering the old object-prior story | active risk | Frame object-prior modules as proposal infrastructure; headline PRISM topology control. |
| Treating COLMAP proxy geometry as ground truth | active risk | Call it sparse scene evidence/proxy, not GT geometry. |
| Treating 2000-iteration public runs as full-budget proof | active risk | Label public rows as medium-budget evidence unless a full-budget run is added. |

## Metric Risks

| risk | status | mitigation |
|---|---|---|
| Mixing training eval and independent render metrics | controlled by M36 | Paper tables use independent `render.py + metrics.py` only. |
| Hiding LPIPS regression on `courtyard` | controlled by M38/M40 | Report LPIPS explicitly and describe tradeoff. |
| Reporting total relaxed attempts as retained edits | controlled by M35 | Use `active_relaxed_commit_count` and validation rollback counts. |

## Dataset Risks

| risk | status | mitigation |
|---|---|---|
| Tanks and Temples mirror lacks sparse tracks | unresolved | Do not use it for geometry-observable claims until COLMAP tracks are rebuilt. |
| Single local parking scene limits generality | partially mitigated | Public `bonsai` and `courtyard` medium runs add cross-scene evidence. |
| Full-budget public-scene result missing | unresolved but not immediate | Run only if final table identifies a concrete missing row. |

## Implementation Risks

| risk | status | mitigation |
|---|---|---|
| Post-commit sync hides all candidates | diagnosed in M34 | Use default-off relaxed refresh only behind strict retained controls. |
| Relaxed edits survive local gate but fail recovery | diagnosed in M35 | Keep recovery validation and rollback records mandatory. |
| Conservative cap under-prunes easy scenes | known tradeoff | Report cap explicitly; future sweep may test higher caps. |

## Current Go/No-Go

Full-budget public Stage35 run: `NO_GO_FOR_NOW`.

Reason: the manuscript still needs figure/citation polish; no concrete table gap currently justifies GPU time.

