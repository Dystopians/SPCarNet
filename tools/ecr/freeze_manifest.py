#!/usr/bin/env python
"""Pre-submission FREEZE manifest: one traceable record of code state,
configs, seeds, environment, and every packed artifact hash."""
import hashlib
import json
import os
import subprocess
import sys

REPO = "/data/peilincai/mesh-splatting"
G1 = "/data/peilincai/gems_stage1"


def sha(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit) if limit else f.read())
    return h.hexdigest()


def main():
    sys.path.insert(0, REPO)
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  text=True).strip()
    import torch
    from tools.ecr.train_fusion import TRAIN_CONFIG
    from tools.ecr.build_cache import TRANSPORT_CONFIG, ALPHA_CALIBRATION
    garden_man = json.load(open(
        f"{G1}/ecr_cache/garden_cleanfixed30k_l4routed/manifest.json"))

    lines = [
        "# FREEZE MANIFEST — pre-submission code/config/seed freeze",
        f"\n- git commit: `{git}` (branch neurips-meshsplatopt-repair)",
        f"- python: {sys.version.split()[0]}  · torch: {torch.__version__}",
        "- env: /home/peilincai/micromamba/envs/mesh_splatting (frozen; "
        "R1/IBR/Difix cells use layered venvs, never modifying it)",
        "\n## Seeds (universal)",
        "- ALL training/eval/bootstrap seeds = 0 (train.py --seed 0; "
        "fusion trainer TRAIN_CONFIG['seed']=0; bootstrap seed 0, 10k; "
        "toy generator --seed 0; problem sampler SEED=0).",
        "- Deterministic caveat (documented in EDIT_AWARE_ECR_PROTOCOL "
        "amendment 4): renders are deterministic within a process, not "
        "across processes; all paired CIs are within-run.",
        "\n## Frozen configs (sha256 of sorted JSON)",
        f"- ELA/PJ-2026 transport config: "
        f"`{hashlib.sha256(json.dumps(TRANSPORT_CONFIG, sort_keys=True).encode()).hexdigest()[:16]}` "
        f"= {json.dumps(TRANSPORT_CONFIG, sort_keys=True)}",
        f"- alpha calibration policy: "
        f"{json.dumps(ALPHA_CALIBRATION, sort_keys=True)}",
        f"- fusion trainer: {json.dumps(TRAIN_CONFIG, sort_keys=True)}",
        f"- final-stack transport kwargs hash (banked garden routed row): "
        f"`{garden_man['transport'].get('fusion_net_sha256','')[:16]}` (net sha) — "
        "per-row config hashes live in each metrics.json ecr block.",
        "\n## Key checkpoint fingerprints (sha256 first 16 MiB)",
    ]
    for name in ("garden_cleanfixed30k", "toy_parking_clean30k",
                 "bonsai_cleanfixed30k"):
        p = (f"{G1}/models/{name}/point_cloud/iteration_30000/"
             f"point_cloud_state_dict.pt")
        lines.append(f"- {name}: `{sha(p, 16 * 1024 * 1024)[:32]}…`")
    lines += [
        "\n## Evidence pack",
        "- RESULTS/STAGE4_ECR/sha256_manifest.txt is the authoritative "
        "per-artifact hash list (byte-verified at fold time); regenerate "
        "with tools/ecr/fold_pack.sh.",
        "- Paper tables: RESULTS/tables_tex/ (regenerate: "
        "tools/analysis/paper_tables.py). Figures: RESULTS/figures/"
        "{ecr_paper,ecr_qual,edit_aware}/ (regenerate: plot_ladder.py, "
        "plot_rd.py, ecr_qual_grids.py, edit_grids.py).",
        "- Reference map: docs/PUBLIC_REFERENCE_MAP.md.",
    ]
    out = os.path.join(REPO, "RESULTS", "STAGE4_ECR", "FREEZE_MANIFEST.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
