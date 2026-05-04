# MeshSplatOpt Related Work Notes

## Neural Point/Gaussian/Mesh Splatting

Position MeshSplatOpt as a post-training scene optimizer for splat/mesh-splat representations. The key distinction is not a new renderer; it is certified topology editing and recovery after a strong renderer has already produced an overcomplete scene.

## Mesh Simplification

Classical simplification and QEM reduce topology but do not optimize against multi-view render loss, sparse COLMAP geometry, or counterfactual rollback gates. The paper should include posthoc simplification as a required baseline, because reviewers will ask whether MeshSplatOpt is just decimation plus finetuning.

## Mesh Repair And Hole Filling

Classical mesh repair assumes coherent connectivity and boundary loops. Real Mesh Splatting checkpoints are effectively triangle soup, so edge-loop CSEF was rejected. This supports the pivot toward spatial/raster evidence and compact-recovery rather than claiming mature real-hole repair.

## Geometry-Regularized View Synthesis

Sparse COLMAP depth is load-bearing in this project. The paper should explicitly cite geometry-regularized rendering and treat sparse-depth guidance as part of the recovery recipe, then ablate it fairly.

## Counterfactual Edit Validation

MeshSplatOpt's rollback/certification story is related to safe model editing and validation-by-rendering. The method checks whether a local topology change preserves downstream render and sparse geometry before it is accepted.

## Negative Results To Include

- teacher distillation from weak ultra-low-topology checkpoints failed;
- direct LPIPS optimization failed on parking;
- fill/snap safety does not imply final quality improvement;
- longer fixed-topology continuation can degrade the compact row;
- aggressive 70 percent public-scene compaction can fail SSIM.

