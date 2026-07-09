"""GEMS Stage-4 — Evidence-Cached Rendering (ECR) package.

PROTOCOL v1.2.0 §4E. The shipped artifact per scene is the triple
{checkpoint + evidence cache + transport renderer}. This package holds:

 - build_cache.py  — builds the per-(checkpoint, scene) evidence cache from
                     TRAIN views only (renders, GT copies, median depths,
                     camera index) and freezes the transport config —
                     including the train-only leave-one-out alpha
                     calibration — into manifest.json.
 - renderer.py     — EcrRenderer: applies the frozen Phase-J transport
                     (utils.evidence_lumigraph_adapter.adapt_frame) to a
                     test view's BASE render+depth handed over as in-memory
                     tensors. Disk reads are confined to the cache root and
                     logged; no GT-bearing Camera object ever crosses into
                     this package (D4, audited by tools/audit_test_path.py
                     --ecr).

Purity boundary (D4, Stage-4 redefinition): train-view images / renders /
depths / cameras are LEGAL render-time inputs (they are part of the shipped
artifact); test-view ground truth remains absolutely forbidden anywhere in
the render path.
"""
