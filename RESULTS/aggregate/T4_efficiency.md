# T4 — Efficiency (E4-EFF)

_generated 2026-07-03T22:28:25.219270+00:00 by tools/gems/report/tables.py — every number computed from metrics.json-derived artifacts; none hand-typed._

> Second-resolution FPS column: bench-only, non-protocol resolution (0.5x linear protocol res); loop: identical to run_eval.py: 3 warmup renders, median of 3 full test-set forward passes, no image I/O; GPU NVIDIA RTX 6000 Ada Generation. These half-res numbers are for the E4 efficiency table only ('bench-only, non-protocol resolution') — no quality metric was computed at this resolution.
> HONESTY CAVEAT on FPS columns: the protocol-res FPS in each metrics.json was measured at eval time (idle GPUs, various same-model RTX 6000 Ada devices); the half-res bench ran on GPU 4 WITH a background process from another user (~21% util at launch). Method-vs-method comparisons WITHIN the half-res column share GPU state (paired-ish); protocol-vs-half-res comparisons across columns are indicative only. Several large scenes show half-res FPS close to or below protocol-res FPS — consistent with triangle-sort-bound rendering (cost dominated by primitive count, not pixels) plus contention; do not read that cross-column delta as a resolution scaling law.
> Pipeline overhead = evidence+prune+finetune wall-clock from row.json stage stamps (measured). 30k-train reference: MEASURED for SS3DM towns (supervised job files, appendix below). For M360/toy/courtyard the original 30k trainings predate the job supervisor: LEDGER GOAL#002 estimates 40-80 min/scene (M360, images_4/2) and GOAL#004 measured ~17 min (toy); those are ESTIMATES, labeled as such — the overhead column stays measured either way.
> Overhead rows exist only for pipeline rows with row.json (B5/B4/B2 tags s2/e1b/e1v2...); B4 overhead is evidence+prune only (no FT).
> Measured 30k-train wall-clocks (supervised jobs): b0_ss3dm_town01=31.2 min (exit 0); b0_ss3dm_town02=32.5 min (exit 0); b0_ss3dm_town03=32.6 min (exit 0); b0_ss3dm_town06=29.5 min (exit 0); b0_toy_occl=23.8 min (exit 0); b0_toy_v2=22.8 min (exit 0)

| suite | scene | method | budget | triangles | disk MB | peak VRAM MB | FPS @protocol res | FPS @0.5x res (bench-only) | prune+FT overhead [min] | overhead vs 30k-train | eval row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-REND | bicycle | B0 | B100 | 9422930 | 908 | 6170 | 38.6 | 31.1 | — | — | bicycle_clean30k_v1 |
| S-REND | bicycle | B5 | B50 | 4711465 | 684 | 4615 | 49.6 | 52.1 | 12.7 | see note (no measured 30k train for this scene) | bicycle_B50_importance_ft_s2 |
| S-REND | bicycle | B4 | B50 | 4711465 | 684 | 4615 | 49.1 | — | 0.4 | see note (no measured 30k train for this scene) | bicycle_B50_importance_noft_s2 |
| S-REND | bicycle | B5 | B25 | 2355732 | 490 | 3573 | 63.5 | 71.9 | 9.9 | see note (no measured 30k train for this scene) | bicycle_B25_importance_ft_s2 |
| S-REND | bicycle | B4 | B25 | 2355732 | 490 | 3573 | 62.4 | — | 0.4 | see note (no measured 30k train for this scene) | bicycle_B25_importance_noft_s2 |
| S-REND | bicycle | B5 | B12.5 | 1177866 | 333 | 2891 | 75.6 | 123.6 | 8.2 | see note (no measured 30k train for this scene) | bicycle_B12_importance_ft_s2 |
| S-REND | bicycle | B4 | B12.5 | 1177866 | 333 | 2891 | 77.2 | — | 0.3 | see note (no measured 30k train for this scene) | bicycle_B12_importance_noft_s2 |
| S-REND | flowers | B0 | B100 | 9649601 | 936 | 6540 | 37.0 | 35.3 | — | — | flowers_clean30k_v1 |
| S-REND | flowers | B5 | B50 | 4824800 | 700 | 4835 | 47.5 | 48.9 | 12.7 | see note (no measured 30k train for this scene) | flowers_B50_importance_ft_s2 |
| S-REND | flowers | B4 | B50 | 4824800 | 700 | 4835 | 47.7 | — | 0.3 | see note (no measured 30k train for this scene) | flowers_B50_importance_noft_s2 |
| S-REND | flowers | B5 | B25 | 2412400 | 491 | 3702 | 61.5 | 70.8 | 9.9 | see note (no measured 30k train for this scene) | flowers_B25_importance_ft_s2 |
| S-REND | flowers | B4 | B25 | 2412400 | 491 | 3702 | 60.7 | — | 0.3 | see note (no measured 30k train for this scene) | flowers_B25_importance_noft_s2 |
| S-REND | flowers | B5 | B12.5 | 1206200 | 328 | 2973 | 73.8 | 122.5 | 8.2 | see note (no measured 30k train for this scene) | flowers_B12_importance_ft_s2 |
| S-REND | flowers | B4 | B12.5 | 1206200 | 328 | 2973 | 73.1 | — | 0.3 | see note (no measured 30k train for this scene) | flowers_B12_importance_noft_s2 |
| S-REND | garden | B0 | B100 | 11568056 | 942 | 6946 | 32.3 | 26.6 | — | — | garden_clean30k_v2 |
| S-REND | garden | B5 | B50 | 5784028 | 735 | 5158 | 41.7 | 38.9 | 13.0 | see note (no measured 30k train for this scene) | garden_B50_importance_ft_e1v2 |
| S-REND | garden | B4 | B50 | 5784028 | 735 | 5158 | 42.2 | — | 0.4 | see note (no measured 30k train for this scene) | garden_B50_importance_noft_e1b |
| S-REND | garden | B3 | B50 | 5784028 | 328 | 4298 | 33.2 | — | — | — | garden_B50_qem_ft_b3 |
| S-REND | garden | B5 | B25 | 2892014 | 555 | 4010 | 54.3 | 54.5 | 10.8 | see note (no measured 30k train for this scene) | garden_B25_importance_ft_e1v2 |
| S-REND | garden | B4 | B25 | 2892014 | 555 | 4010 | 53.8 | — | 0.4 | see note (no measured 30k train for this scene) | garden_B25_importance_noft_e1b |
| S-REND | garden | B5 | B12.5 | 1446007 | 390 | 3231 | 65.9 | 110.1 | 9.5 | see note (no measured 30k train for this scene) | garden_B12_importance_ft_s2 |
| S-REND | garden | B4 | B12.5 | 1446007 | 390 | 3231 | 63.6 | — | 0.4 | see note (no measured 30k train for this scene) | garden_B12_importance_noft_s2 |
| S-REND | stump | B0 | B100 | 9277087 | 918 | 6374 | 37.2 | 30.5 | — | — | stump_clean30k_v1 |
| S-REND | stump | B5 | B50 | 4638543 | 678 | 4675 | 49.7 | 48.1 | 12.1 | see note (no measured 30k train for this scene) | stump_B50_importance_ft_s2 |
| S-REND | stump | B4 | B50 | 4638543 | 678 | 4675 | 49.7 | — | 0.3 | see note (no measured 30k train for this scene) | stump_B50_importance_noft_s2 |
| S-REND | stump | B5 | B25 | 2319271 | 490 | 3603 | 61.6 | 72.6 | 9.8 | see note (no measured 30k train for this scene) | stump_B25_importance_ft_s2 |
| S-REND | stump | B4 | B25 | 2319271 | 490 | 3603 | 62.7 | — | 0.3 | see note (no measured 30k train for this scene) | stump_B25_importance_noft_s2 |
| S-REND | stump | B5 | B12.5 | 1159635 | 338 | 2917 | 75.9 | 125.7 | 8.3 | see note (no measured 30k train for this scene) | stump_B12_importance_ft_s2 |
| S-REND | stump | B4 | B12.5 | 1159635 | 338 | 2917 | 75.9 | — | 0.3 | see note (no measured 30k train for this scene) | stump_B12_importance_noft_s2 |
| S-REND | treehill | B0 | B100 | 9527637 | 934 | 6349 | 35.5 | 30.0 | — | — | treehill_clean30k_v1 |
| S-REND | treehill | B5 | B50 | 4763818 | 701 | 4704 | 48.0 | 55.7 | 12.3 | see note (no measured 30k train for this scene) | treehill_B50_importance_ft_s2 |
| S-REND | treehill | B4 | B50 | 4763818 | 701 | 4704 | 48.1 | — | 0.3 | see note (no measured 30k train for this scene) | treehill_B50_importance_noft_s2 |
| S-REND | treehill | B5 | B25 | 2381909 | 493 | 3640 | 60.7 | 73.5 | 9.8 | see note (no measured 30k train for this scene) | treehill_B25_importance_ft_s2 |
| S-REND | treehill | B4 | B25 | 2381909 | 493 | 3640 | 60.6 | — | 0.3 | see note (no measured 30k train for this scene) | treehill_B25_importance_noft_s2 |
| S-REND | treehill | B5 | B12.5 | 1190954 | 328 | 2946 | 74.8 | 124.2 | 8.2 | see note (no measured 30k train for this scene) | treehill_B12_importance_ft_s2 |
| S-REND | treehill | B4 | B12.5 | 1190954 | 328 | 2946 | 74.0 | — | 0.3 | see note (no measured 30k train for this scene) | treehill_B12_importance_noft_s2 |
| S-REND | room | B0 | B100 | 11173063 | 819 | 7885 | 31.5 | 30.8 | — | — | room_clean30k_v1 |
| S-REND | room | B5 | B50 | 5586531 | 637 | 6063 | 39.0 | 43.0 | 16.3 | see note (no measured 30k train for this scene) | room_B50_importance_ft_s2 |
| S-REND | room | B4 | B50 | 5586531 | 637 | 6063 | 38.7 | — | 0.6 | see note (no measured 30k train for this scene) | room_B50_importance_noft_s2 |
| S-REND | room | B5 | B25 | 2793265 | 492 | 4939 | 45.6 | 61.1 | 13.3 | see note (no measured 30k train for this scene) | room_B25_importance_ft_s2 |
| S-REND | room | B4 | B25 | 2793265 | 492 | 4939 | 46.3 | — | 0.7 | see note (no measured 30k train for this scene) | room_B25_importance_noft_s2 |
| S-REND | room | B5 | B12.5 | 1396632 | 348 | 4168 | 54.0 | 92.6 | 11.4 | see note (no measured 30k train for this scene) | room_B12_importance_ft_s2 |
| S-REND | room | B4 | B12.5 | 1396632 | 348 | 4168 | 53.7 | — | 0.6 | see note (no measured 30k train for this scene) | room_B12_importance_noft_s2 |
| S-REND | counter | B0 | B100 | 9850919 | 729 | 8954 | 24.5 | 26.2 | — | — | counter_clean30k_v1 |
| S-REND | counter | B5 | B50 | 4925459 | 556 | 6706 | 31.9 | 38.5 | 16.2 | see note (no measured 30k train for this scene) | counter_B50_importance_ft_s2 |
| S-REND | counter | B4 | B50 | 4925459 | 556 | 6706 | 31.9 | — | 0.5 | see note (no measured 30k train for this scene) | counter_B50_importance_noft_s2 |
| S-REND | counter | B5 | B25 | 2462729 | 418 | 5349 | 40.1 | 57.2 | 13.1 | see note (no measured 30k train for this scene) | counter_B25_importance_ft_s2 |
| S-REND | counter | B4 | B25 | 2462729 | 418 | 5349 | 40.1 | — | 0.5 | see note (no measured 30k train for this scene) | counter_B25_importance_noft_s2 |
| S-REND | counter | B5 | B12.5 | 1231364 | 291 | 4447 | 48.0 | 79.5 | 11.7 | see note (no measured 30k train for this scene) | counter_B12_importance_ft_s2 |
| S-REND | counter | B4 | B12.5 | 1231364 | 291 | 4447 | 48.0 | — | 0.4 | see note (no measured 30k train for this scene) | counter_B12_importance_noft_s2 |
| S-REND | kitchen | B0 | B100 | 9716239 | 709 | 7878 | 26.4 | 26.1 | — | — | kitchen_clean30k_v1 |
| S-REND | kitchen | B5 | B50 | 4858119 | 527 | 6033 | 33.8 | 40.0 | 16.8 | see note (no measured 30k train for this scene) | kitchen_B50_importance_ft_s2 |
| S-REND | kitchen | B4 | B50 | 4858119 | 527 | 6033 | 33.3 | — | 0.6 | see note (no measured 30k train for this scene) | kitchen_B50_importance_noft_s2 |
| S-REND | kitchen | B5 | B25 | 2429059 | 392 | 4869 | 42.0 | 60.5 | 13.4 | see note (no measured 30k train for this scene) | kitchen_B25_importance_ft_s2 |
| S-REND | kitchen | B4 | B25 | 2429059 | 392 | 4869 | 41.7 | — | 0.6 | see note (no measured 30k train for this scene) | kitchen_B25_importance_noft_s2 |
| S-REND | kitchen | B5 | B12.5 | 1214529 | 279 | 4115 | 50.7 | 81.3 | 11.3 | see note (no measured 30k train for this scene) | kitchen_B12_importance_ft_s2 |
| S-REND | kitchen | B4 | B12.5 | 1214529 | 279 | 4115 | 50.8 | — | 0.5 | see note (no measured 30k train for this scene) | kitchen_B12_importance_noft_s2 |
| S-REND | bonsai | B0 | B100 | 10834182 | 924 | 7824 | 29.5 | 26.3 | — | — | bonsai_clean30k_v1 |
| S-REND | bonsai | B5 | B50 | 5417091 | 707 | 6084 | 36.8 | 40.7 | 16.3 | see note (no measured 30k train for this scene) | bonsai_B50_importance_ft_s2 |
| S-REND | bonsai | B4 | B50 | 5417091 | 707 | 6084 | 36.9 | — | 0.6 | see note (no measured 30k train for this scene) | bonsai_B50_importance_noft_s2 |
| S-REND | bonsai | B5 | B25 | 2708545 | 530 | 4973 | 43.6 | 56.6 | 13.3 | see note (no measured 30k train for this scene) | bonsai_B25_importance_ft_s2 |
| S-REND | bonsai | B4 | B25 | 2708545 | 530 | 4973 | 44.5 | — | 0.6 | see note (no measured 30k train for this scene) | bonsai_B25_importance_noft_s2 |
| S-REND | bonsai | B5 | B12.5 | 1354272 | 369 | 4206 | 52.3 | 92.7 | 11.7 | see note (no measured 30k train for this scene) | bonsai_B12_importance_ft_s2 |
| S-REND | bonsai | B4 | B12.5 | 1354272 | 369 | 4206 | 52.1 | — | 0.5 | see note (no measured 30k train for this scene) | bonsai_B12_importance_noft_s2 |
| S-GEO | ss3dm_town01 | B0 | B100 | 12054155 | 942 | 5312 | 53.2 | 40.7 | — | — | ss3dm_town01_clean30k_geo_v1 |
| S-GEO | ss3dm_town01 | B5 | B50 | 6027077 | 735 | 3741 | 70.1 | 57.1 | 12.9 | 41% of measured 31 min | ss3dm_town01_B50_geo_v1 |
| S-GEO | ss3dm_town01 | B4 | B50 | 6027077 | 735 | 3741 | 33.4 | — | 1.4 | 4% of measured 31 min | ss3dm_town01_B50_importance_noft_s2 |
| S-GEO | ss3dm_town01 | B5 | B25 | 3013538 | 541 | 2745 | 86.8 | — | 9.5 | 30% of measured 31 min | ss3dm_town01_B25_importance_ft_s2 |
| S-GEO | ss3dm_town01 | B4 | B25 | 3013538 | 541 | 2745 | 88.0 | — | 0.9 | 3% of measured 31 min | ss3dm_town01_B25_importance_noft_s2 |
| S-GEO | ss3dm_town02 | B0 | B100 | 15573225 | 1022 | 6163 | 48.0 | 36.4 | — | — | ss3dm_town02_clean30k_geo_v1 |
| S-GEO | ss3dm_town02 | B5 | B50 | 7786612 | 792 | 4215 | 64.2 | 53.4 | 13.3 | 41% of measured 33 min | ss3dm_town02_B50_geo_v1 |
| S-GEO | ss3dm_town02 | B4 | B50 | 7786612 | 792 | 4215 | 63.3 | — | 1.1 | 3% of measured 33 min | ss3dm_town02_B50_importance_noft_s2 |
| S-GEO | ss3dm_town02 | B5 | B25 | 3893306 | 595 | 3035 | 81.4 | — | 10.4 | 32% of measured 33 min | ss3dm_town02_B25_importance_ft_s2 |
| S-GEO | ss3dm_town02 | B4 | B25 | 3893306 | 595 | 3035 | 78.9 | — | 1.0 | 3% of measured 33 min | ss3dm_town02_B25_importance_noft_s2 |
| S-GEO | ss3dm_town03 | B0 | B100 | 15148441 | 1009 | 6018 | 50.0 | 36.7 | — | — | ss3dm_town03_clean30k_geo_v1 |
| S-GEO | ss3dm_town03 | B5 | B50 | 7574220 | 781 | 4125 | 65.0 | 52.9 | 13.2 | 41% of measured 33 min | ss3dm_town03_B50_geo_v1 |
| S-GEO | ss3dm_town03 | B4 | B50 | 7574220 | 781 | 4125 | 37.0 | — | 1.2 | 4% of measured 33 min | ss3dm_town03_B50_importance_noft_s2 |
| S-GEO | ss3dm_town03 | B5 | B25 | 3787110 | 589 | 2988 | 82.8 | — | 9.5 | 29% of measured 33 min | ss3dm_town03_B25_importance_ft_s2 |
| S-GEO | ss3dm_town03 | B4 | B25 | 3787110 | 589 | 2988 | 83.9 | — | 1.0 | 3% of measured 33 min | ss3dm_town03_B25_importance_noft_s2 |
| S-GEO | ss3dm_town06 | B0 | B100 | 8921032 | 818 | 4424 | 63.9 | 51.0 | — | — | ss3dm_town06_clean30k_geo_v1 |
| S-GEO | ss3dm_town06 | B5 | B50 | 4460516 | 636 | 3214 | 80.3 | 66.9 | 11.6 | 39% of measured 29 min | ss3dm_town06_B50_geo_v1 |
| S-GEO | ss3dm_town06 | B4 | B50 | 4460516 | 636 | 3214 | 80.2 | — | 1.8 | 6% of measured 29 min | ss3dm_town06_B50_importance_noft_s2 |
| S-GEO | ss3dm_town06 | B5 | B25 | 2230258 | 453 | 2389 | 100.1 | — | 8.3 | 28% of measured 29 min | ss3dm_town06_B25_importance_ft_s2 |
| S-GEO | ss3dm_town06 | B4 | B25 | 2230258 | 453 | 2389 | 100.7 | — | 0.8 | 3% of measured 29 min | ss3dm_town06_B25_importance_noft_s2 |
| S-DEV | toy_parking | B0 | B100 | 6590559 | 542 | 4189 | 62.3 | 74.2 | — | — | toy_parking_clean30k_v1 |
| S-DEV | toy_parking | B5 | B50 | 3295279 | 426 | 3197 | 75.5 | 115.5 | 7.6 | see note (no measured 30k train for this scene) | toy_parking_B50_importance_ft_e1v2 |
| S-DEV | toy_parking | B4 | B50 | 3295279 | 426 | 3197 | 76.1 | — | 0.2 | see note (no measured 30k train for this scene) | toy_parking_B50_importance_noft_e1b |
| S-DEV | toy_parking | B3 | B50 | 3295275 | 209 | 2692 | 36.5 | — | — | — | toy_parking_B50_qem_ft_b3 |
| S-DEV | toy_parking | B5 | B25 | 1647639 | 320 | 2549 | 90.0 | — | 6.6 | see note (no measured 30k train for this scene) | toy_parking_B25_importance_ft_e1v2 |
| S-DEV | toy_parking | B4 | B25 | 1647639 | 320 | 2549 | 90.2 | — | 0.2 | see note (no measured 30k train for this scene) | toy_parking_B25_importance_noft_e1b |
| S-DEV | toy_parking_v2 | B0 | B100 | 6665411 | 549 | 4266 | 61.0 | — | — | — | toy_parking_v2_clean30k_v1 |
| S-DEV | toy_parking_v2 | B5 | B50 | 3332705 | 431 | 3248 | 74.4 | — | 8.2 | see note (no measured 30k train for this scene) | toy_parking_v2_B50_importance_ft_s2 |
| S-DEV | toy_parking_v2 | B4 | B50 | 3332705 | 431 | 3248 | 73.7 | — | 0.3 | see note (no measured 30k train for this scene) | toy_parking_v2_B50_importance_noft_s2 |
| S-DEV | toy_parking_occl | B0 | B100 | 6659597 | 554 | 4305 | 60.5 | — | — | — | toy_parking_occl_clean30k_v1 |
| S-DEV | toy_parking_occl | B5 | B50 | 3329798 | 436 | 3276 | 74.2 | — | 8.2 | see note (no measured 30k train for this scene) | toy_parking_occl_B50_importance_ft_s2 |
| S-DEV | toy_parking_occl | B4 | B50 | 3329798 | 436 | 3276 | 75.0 | — | 0.3 | see note (no measured 30k train for this scene) | toy_parking_occl_B50_importance_noft_s2 |
| S-DEV | courtyard | B0 | B100 | 4374929 | 463 | 2544 | 104.9 | 101.6 | — | — | courtyard_clean30k_v4 |
| S-DEV | courtyard | B5 | B50 | 2187464 | 355 | 1910 | 122.7 | 126.0 | 5.9 | see note (no measured 30k train for this scene) | courtyard_B50_ft_v4 |
| S-DEV | courtyard | B4 | B50 | 2187464 | 355 | 1910 | 123.1 | — | 0.8 | see note (no measured 30k train for this scene) | courtyard_B50_importance_noft_v4 |
| S-DEV | courtyard | B3 | B50 | 2187463 | 187 | 1549 | 148.7 | — | — | — | courtyard_B50_qem_ft_b3 |
| S-DEV | courtyard | B5 | B25 | 1093732 | 248 | 1468 | 140.3 | — | 5.3 | see note (no measured 30k train for this scene) | courtyard_B25_importance_ft_e1v2 |
| S-DEV | courtyard | B4 | B25 | 1093732 | 248 | 1468 | 141.0 | — | 0.4 | see note (no measured 30k train for this scene) | courtyard_B25_importance_noft_v4 |

---

## CONTEXT appendix (NOT corpus rows — no CIs vs GEMS)

### R1 — 3DGS + storage-matched opacity-prune reference (LEDGER GOAL#017; sanctioned outside-single-mouth exception; full table: `analysis/r1_3dgs_reference/r1_table.md`)

Context only per Stage2 prompt section 4: NOT a claim target; NON-CLAIMS already disclaim SOTA novel-view quality vs the 3DGS family. Same llff8 splits/resolutions (name-asserted); disk MB = each representation's shippable artifact.

| scene | method | primitives | disk MB | FPS | peak VRAM MB |
|---|---|---|---|---|---|
| garden | 3DGS 30k (vanilla) | 4,158,575 gaussians | 983.6 | 100.8 | 5610 |
| garden | 3DGS opacity-prune+FT5k (storage-matched) | 3,106,001 gaussians | 734.6 | 140.3 | 4965 |
| garden | GEMS B0 (corpus row, quoted) | 11,568,056 triangles | 942.0 | 32.3 | — |
| garden | GEMS B5@B50 (corpus row, quoted) | 5,784,028 triangles | 734.6 | 41.7 | — |
| bicycle | 3DGS 30k (vanilla) | 4,925,145 gaussians | 1164.9 | 79.4 | 6164 |
| bicycle | 3DGS opacity-prune+FT5k (storage-matched) | 2,892,207 gaussians | 684.0 | 156.2 | 4859 |
| bicycle | GEMS B0 (corpus row, quoted) | 9,422,930 triangles | 908.1 | 38.6 | — |
| bicycle | GEMS B5@B50 (corpus row, quoted) | 4,711,465 triangles | 684.0 | 49.6 | — |
| kitchen | 3DGS 30k (vanilla) | 1,592,262 gaussians | 376.6 | 149.4 | 8152 |
| kitchen | 3DGS opacity-prune+FT5k (storage-matched) | 1,592,262 gaussians | 376.6 | 149.4 | 8152 |
| kitchen | GEMS B0 (corpus row, quoted) | 9,716,239 triangles | 708.7 | 26.4 | — |
| kitchen | GEMS B5@B50 (corpus row, quoted) | 4,858,119 triangles | 527.3 | 33.8 | — |

A3 positioning (verbatim conclusion lives in r1_table.md): at matched artifact storage 3DGS renders these scenes 2.1–3.4 dB above GEMS B5@B50 at 3.1–4.4x FPS; GEMS's deliverables (mesh artifact, g1–g4/downstream consumability, preservation-exactness, 50%-reduction-at-iso) have no 3DGS-family equivalent artifact. R1 contextualizes, gates nothing.
