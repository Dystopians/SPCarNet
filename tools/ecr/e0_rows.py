#!/usr/bin/env python
"""GEMS Stage-4 M-E0 row table (single source of truth for the E0 chains).

Emits, for --base {primary,b50} and optional --scene filter, the shell
commands for: cache build -> run_eval --renderer ecr -> audit --ecr.
Pre-registered in LEDGER GOAL #E-00; one frozen transport config everywhere.
"""
import argparse
import os

PY = "/home/peilincai/micromamba/envs/mesh_splatting/bin/python"
REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"
FULL9 = ["garden", "bicycle", "flowers", "stump", "treehill",
         "room", "counter", "kitchen", "bonsai"]
TOWNS = ["ss3dm_town01", "ss3dm_town02", "ss3dm_town03", "ss3dm_town06"]
ALL_SCENES = FULL9 + TOWNS + ["toy_parking"]


def base_checkpoint(scene: str, base: str) -> tuple[str, str]:
    """(checkpoint path, base tag). PRIMARY anchor: cleanfixed30k on full9,
    clean30k on SS3DM/toy (no separate fixed checkpoint exists there — the
    clean30k IS the primary anchor, LEDGER #R-01 scope)."""
    if base == "primary":
        if scene in FULL9:
            run, it, tag = f"{scene}_cleanfixed30k", 30000, "cleanfixed30k"
        else:
            run, it, tag = f"{scene}_clean30k", 30000, "clean30k"
    elif base == "b50":
        legacy = {"garden", "toy_parking"}
        suffix = "e1v2" if scene in legacy else "s2"
        run, it, tag = f"{scene}_B50_importance_ft_{suffix}", 40000, "B50"
    else:
        raise ValueError(base)
    ckpt = os.path.join(G1, "models", run, "point_cloud",
                        f"iteration_{it}", "point_cloud_state_dict.pt")
    return ckpt, tag


def emit(scene: str, base: str, gpu: int) -> str:
    ckpt, tag = base_checkpoint(scene, base)
    cache = os.path.join(G1, "ecr_cache", f"{scene}_{tag}")
    row = os.path.join(G1, "eval", f"e0_{scene}_{tag}_pj2026_v1")
    audit = os.path.join(G1, "eval", f"e0_{scene}_{tag}_pj2026_audit")
    lines = [
        f"echo '=== E0 {scene} {tag} ==='",
        f"test -f {ckpt} || {{ echo 'MISSING CKPT {ckpt}'; exit 3; }}",
        f"{PY} -m tools.ecr.build_cache --checkpoint {ckpt} --scene {scene} "
        f"--out {cache} --gpu {gpu}",
        f"{PY} run_eval.py --checkpoint {ckpt} --scene {scene} --out {row} "
        f"--gpu {gpu} --renderer ecr --ecr-cache {cache}",
        f"CUDA_VISIBLE_DEVICES={gpu} {PY} tools/audit_test_path.py "
        f"--checkpoint {ckpt} --scene {scene} --out {audit} "
        f"--ecr --ecr-cache {cache} --fast",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=("primary", "b50"), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    scenes = args.scenes or ALL_SCENES
    print("#!/bin/bash\nset -e")
    print(f"cd {REPO}")
    for scene in scenes:
        print(emit(scene, args.base, args.gpu))


if __name__ == "__main__":
    main()
