# EXP-TEMP (GOAL #E-13) — temporal / view-path stability

GT-free deterministic camera paths (Catmull-Rom + slerp through the name-ordered test poses, 120 frames); roughness = mean|I_t − I_{t−1}|; acceptance bar: final/base ratio ≤ 1.5.

| scene | base rough. (mean/P95) | final rough. (mean/P95) | ratio (mean/P95) | support switches/step (mean/max) | verdict |
|---|---|---|---|---|---|
| garden | 0.1408/0.1721 | 0.1391/0.1692 | **0.988**/0.983 | 3.2/4 | PASS |
| bonsai | 0.1286/0.1701 | 0.1288/0.1717 | **1.002**/1.009 | 1.5/2 | PASS |
| ss3dm_town01 | 0.0616/0.1365 | 0.0635/0.1385 | **1.030**/1.015 | 2.0/2 | PASS |

**Reading:** despite per-view-independent transport and ~2–3 support-set switches per step, path roughness is 0.99–1.03× the base renderer's own level (garden is actually SMOOTHER than base — the transport suppresses base rendering noise). The structural β·valid gate changes support smoothly; no popping is visible in the videos (<scene>_path.mp4, side-by-side base | final | β).
