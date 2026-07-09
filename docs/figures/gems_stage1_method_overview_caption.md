# GEMS Stage-One Method Figure Caption

**Figure. GEMS method overview.** Starting from a clean MeshSplatting checkpoint and training views, GEMS first renders only the training trajectory to build per-triangle evidence scores. A budget engine then keeps the highest-evidence triangles under a target triangle budget and removes low-evidence primitives. The compact checkpoint is repaired by topology-frozen feature-only fine-tuning, which keeps positions and opacity/weights fixed to avoid the position-drift failure observed in resumed MeshSplatting optimization. The resulting model remains a plain MeshSplatting checkpoint and is evaluated through the same pure held-out render path as the baseline. Dashed branches show train-only extensions: pseudo-view teacher distillation and geometry diagnostics for free-space / floater analysis.

Suggested short slide caption:

> GEMS compresses MeshSplatting at the representation level: train-view evidence ranks triangles, the budget engine removes low-evidence primitives, and a safe feature-only repair step recovers appearance without adding test-time modules.

Current honesty note for internal use:

- The validated core is evidence-guided pruning plus feature-only repair.
- Geometry-loss repair is currently at risk after two failed E2 attempts.
- Teacher pseudo-view distillation is pre-registered but not yet validated.
