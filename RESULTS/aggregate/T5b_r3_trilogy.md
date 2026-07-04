# T5b — R3 consumption-closure summary (script-extracted)

## R3.a occupancy routes (GOAL#R-02)

Verdict (verbatim summary.json): `{"toy_per_model_pass": {"toy_parking__clean30k": false, "toy_parking__B50_importance_ft_e1v2_40000": false, "toy_parking__B25_importance_ft_e1v2_40000": false}, "toy_preregistered_bar_PASS": false, "courtyard_per_model_pass_frozen_params": {"courtyard__clean30k": false, "courtyard__B50_importance_ft_e1v2_40000": false}, "note": "bar (pre-registered): >=50% relative false-free reduction at <=2x false-occupied, paired CI excl. 0, on ALL 3 toy models; courtyard reported as frozen-params transfer outcome"}`

| cell | route-i false-free | route-ii false-free | route-i false-occ | route-ii false-occ |
|---|---|---|---|---|
| toy_parking__clean30k | 0.5895 | 0.8315 | 0.0305 | 0.0150 |
| toy_parking__B50_importance_ft_e1v2_40000 | 0.5974 | 0.8315 | 0.0296 | 0.0150 |
| toy_parking__B25_importance_ft_e1v2_40000 | 0.6149 | 0.8318 | 0.0278 | 0.0150 |
| courtyard__clean30k | 0.6503 | 0.8388 | 0.1053 | 0.0367 |
| courtyard__B50_importance_ft_e1v2_40000 | 0.6680 | 0.8388 | 0.0958 | 0.0367 |

## R3.c planner loop v0 (GOAL#R-03)

Verdict (verbatim summary.json): `{"P1_preservation_per_comparison_ok": {"toy_parking:P1_B50_importance_ft_e1v2_40000_minus_clean__route_i": true, "toy_parking:P1_B25_importance_ft_e1v2_40000_minus_clean__route_i": true, "courtyard:P1_B50_importance_ft_e1v2_40000_minus_clean__route_i": "unevaluable (n_common_found=0)"}, "P1_PASS": true, "P1_n_unevaluable": 1, "P2_route_ii_more_collisions_per_cell": {"toy_parking:P2_route_ii_minus_i__clean30k": false, "toy_parking:P2_route_ii_minus_i__B50_importance_ft_e1v2_40000": false, "toy_parking:P2_route_ii_minus_i__B25_importance_ft_e1v2_40000": false, "courtyard:P2_route_ii_minus_i__clean30k": "unevaluable (n_common_found=0)", "courtyard:P2_route_ii_minus_i__B50_importance_ft_e1v2_40000": "unevaluable (n_common_found=0)"}, "P2_PASS": false, "P2_n_unevaluable": 2, "note": "P1 bar: no`

| cell | plans found /100 | collisions /100 plans | path inflation vs GTREF |
|---|---|---|---|
| toy_parking__GTREF | 100 | 1.0 | n/a |
| toy_parking__clean30k__route_i | 7 | 0.0 | n/a |
| toy_parking__clean30k__route_ii | 10 | 0.0 | n/a |
| toy_parking__B50_importance_ft_e1v2_40000__route_i | 7 | 0.0 | n/a |
| toy_parking__B50_importance_ft_e1v2_40000__route_ii | 10 | 0.0 | n/a |
| toy_parking__B25_importance_ft_e1v2_40000__route_i | 7 | 0.0 | n/a |
| toy_parking__B25_importance_ft_e1v2_40000__route_ii | 10 | 0.0 | n/a |
| courtyard__GTREF | 100 | 2.0 | n/a |
| courtyard__clean30k__route_i | 0 | n/a | n/a |
| courtyard__clean30k__route_ii | 28 | 10.7 | n/a |
| courtyard__B50_importance_ft_e1v2_40000__route_i | 0 | n/a | n/a |
| courtyard__B50_importance_ft_e1v2_40000__route_ii | 28 | 10.7 | n/a |

## R3.b certified sub-mesh (GOAL#R-06)

Verdict (verbatim summary.json): `{"per_bar_cell": {"toy_parking__clean30k": {"found_rate": 0.14, "found_ok": false, "collisions_per_100": 0.0, "coll_cap": 2.0, "coll_ok": true, "ff_relative_change_vs_raw": 0.16287581112333557, "ff_ok": false, "cell_PASS": false}, "toy_parking__B50_importance_ft_e1v2_40000": {"found_rate": 0.14, "found_ok": false, "collisions_per_100": 0.0, "coll_cap": 2.0, "coll_ok": true, "ff_relative_change_vs_raw": 0.14741786117819072, "ff_ok": false, "cell_PASS": false}, "courtyard__clean30k": {"found_rate": 0.42, "found_ok": true, "collisions_per_100": 16.666666666666668, "coll_cap": 3.0, "coll_ok": false, "ff_relative_change_vs_raw": 0.30545157884633845, "ff_ok": false, "cell_PASS": false}, "courtyard__B50_importance_ft_e1v2_40000": {"found_rate": 0.42, "found_ok": true, "collisions_per_100": 16.666`

| cell | found /100 | coll /100 | d1 ff (sub-mesh) | d1 ff (raw route-i) | kept frac of finite |
|---|---|---|---|---|---|
| toy_parking__clean30k | 14 | 0.0 | 0.6855 | 0.5895 | 0.1137 |
| toy_parking__B50_importance_ft_e1v2_40000 | 14 | 0.0 | 0.6855 | 0.5974 | 0.2274 |
| toy_parking__B25_importance_ft_e1v2_40000 | 14 | 0.0 | 0.6822 | 0.6149 | 0.4566 |
| courtyard__clean30k | 42 | 16.7 | 0.8489 | 0.6503 | 0.0778 |
| courtyard__B50_importance_ft_e1v2_40000 | 42 | 16.7 | 0.8489 | 0.6680 | 0.1556 |

## R3-FINAL V1 three-state visibility carving (GOAL#C-02)

Verdict (verbatim summary.json): `{"mechanism": "V1 log-odds visibility carving", "stage3_r3_final_v1_pass": false, "toy_bar_pass_all_reported": false, "courtyard_fix_target_pass_all_reported": false, "selected_params": {"theta_free": -0.5, "theta_occ": 1.0, "v_min": 1, "r_inf": 1.0}, "note": "PASS requires toy found >=0.5x GTREF at <=3 collisions/100 and median path inflation <=1.5x, plus courtyard >=30/100 found and <=3 collisions/100. If false, Stage3 may consume V2/V3 only if pre-registered as near-miss mechanisms; otherwise write the impossibility addendum after <=3 mechanisms total."}`

| scene | model | FREE frac | UNKNOWN frac | FREE@GT-occ | blocked@GT-free | found /100 | coll /100 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| toy_parking | clean30k | 0.3375 | 0.6462 | 0.0975 | 0.6517 | 0 | n/a | FAIL |
| toy_parking | B50 | 0.3375 | 0.6462 | 0.0975 | 0.6517 | 0 | n/a | FAIL |
| courtyard | clean30k | 0.2259 | 0.7649 | 0.0621 | 0.7649 | 0 | n/a | FAIL |
| courtyard | B50 | 0.2259 | 0.7649 | 0.0621 | 0.7649 | 0 | n/a | FAIL |

**Stage3 closure:** V1 is a hard non-near-miss (0/100 found on toy and courtyard clean/B50). V2/V3 are not launched; `RESULTS/CONSUMPTION_IMPOSSIBILITY.md` closes the axis as IMPOSSIBILITY x4 route families.
