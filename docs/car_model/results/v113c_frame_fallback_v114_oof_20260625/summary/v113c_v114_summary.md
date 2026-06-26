# v113c Frame Fallback and v114 OOF Refit Summary

Date: 2026-06-25

## Garden Metrics

| method | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---|
| clean_meshsplatting | 25.029211 | 0.780035 | 0.201314 | local clean MeshSplatting |
| v106_parent | 25.790945 | 0.799382 | 0.174480 | current quality parent |
| v110b | 25.430321 | 0.783703 | 0.186970 | strict train/even candidate gate |
| v113b_scene_fallback | 25.790945 | 0.799382 | 0.174480 | full scene OOT fallback to parent |
| v113c_frame_fallback | 25.499817 | 0.786888 | 0.184260 | per-frame OOT fallback |

## v113c Conclusion

v113c is a real gate-policy improvement over v110b on garden, but it is not a quality breakthrough over v106. It disables only two out-of-trajectory target frames (`00020.png`, `00021.png`) and reduces mask-weighted OOT exposure from `0.090031` to `0.0`, but final garden metrics remain below the v106 parent.

## v114 Running Line

v114 moves the next attempt to candidate generation: `v114_oof_refit_pod_moe` uses train/all POD-MoE coefficients while out-of-fold gains only cap expert reliability. The garden field build is running under `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden/`.
