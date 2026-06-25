# v90/v91 Process Result Audit

Date: 2026-06-25

Verdict: no v90/v91 run currently provides promotion evidence.

- No active v90/v91 process was found during the main-agent and subagent audits.
- `/dev/shm/peilincai_spcarnet_v90_source_mixture_adaptive_20260625` has only partial logs; no final held-out `results.json` was found.
- `/dev/shm/peilincai_spcarnet_v91_target_debt_support_20260625` was interrupted during target-debt support ranking and has no valid result.
- `/dev/shm/peilincai_spcarnet_v91_target_debt_support_fast_20260625` has partial logs but no completed selection/evaluation and no final held-out metrics.
- Any stale `trainval_gate_results.json` in those roots is not promotion evidence.

Promotion remains blocked until a new representation-baked candidate beats the v84/v86 counter anchor on PSNR, SSIM, and LPIPS, then passes hard-triad/full9.
