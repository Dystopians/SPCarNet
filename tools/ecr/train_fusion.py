#!/usr/bin/env python
"""GEMS Stage-4 L3 trainer: per-scene learned fusion net (GOAL #E-04).

Trains FusionNet on TRAIN views only (leave-one-out transport features ->
per-pixel alpha), FROZEN config across scenes, last iterate = THE model
(no checkpoint selection, D4). Writes fusion_net.pt + fusion_training.json
into the cache dir and rewrites manifest.json with fuse="learned" (keeping
the inner fuse + calibrated K/alpha provenance for the audit trail).

The trainer reads ONLY the evidence cache (renders/gt/depths/cameras) — the
same purity shape as the alpha calibration.

Usage:
    python -m tools.ecr.train_fusion --cache <cache_dir> [--gpu N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frozen L3 training config (mechanism axis 2: loss form; one config, all
# scenes; pre-registered in LEDGER #E-04).
TRAIN_CONFIG = {
    "steps": 3000,
    "batch": 4,
    "crop": 256,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "loss": "l1+0.2*dssim",
    "dssim_weight": 0.2,
    "seed": 0,
    "feature_dtype": "float16",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--routed", action="store_true",
                    help="L4: 12-ch inputs (+fused warped color) and a "
                         "2-ch (alpha, beta) routing head; same frozen "
                         "trainer (GOAL #E-05)")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import torch
    from pathlib import Path
    from utils.evidence_lumigraph_adapter import FrameRecord, load_camera_index
    from utils.loss_utils import ssim
    from tools.ecr.renderer import ConfinedFrameLoader
    from tools.ecr.fusion import (FusionNet, compose_routed,
                                  compute_transport_features,
                                  features_to_input, features_to_input_routed)

    torch.manual_seed(TRAIN_CONFIG["seed"])
    cache = Path(args.cache).resolve()
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transport = dict(manifest["transport"])
    inner_fuse = str(transport.get("fuse", "single"))
    device = torch.device("cuda")

    ext = manifest.get("image_ext", {})
    rext = str(ext.get("renders", "png"))
    gext = str(ext.get("gt", "png"))
    cameras = {c.image_name: c for c in load_camera_index(cache / "camera_index.json")}
    frames = [FrameRecord(idx=i, name=n,
                          render_path=cache / "renders" / f"{n}.{rext}",
                          gt_path=cache / "gt" / f"{n}.{gext}",
                          depth_path=cache / "depths" / f"{n}.npy",
                          camera=cameras[n])
              for i, n in enumerate(manifest["train_views"])]
    loader = ConfinedFrameLoader(cache, device=device, max_cached=16)

    feat_kwargs = dict(
        k=int(transport["k"]),
        residual_clip=float(transport["residual_clip"]),
        min_confidence=float(transport["min_confidence"]),
        depth_abs_tol=float(transport["depth_abs_tol"]),
        depth_rel_tol=float(transport["depth_rel_tol"]),
        direction_weight=float(transport["direction_weight"]),
        fuse=inner_fuse,
        bands=int(transport.get("bands", 4)),
    )

    # ---- stage 1: precompute LOO features for every train view (fp16) ----
    t0 = time.time()
    stack_fn = features_to_input_routed if args.routed else features_to_input
    inputs, targets = [], []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            support = [f for f in frames if f.name != frame.name]
            feats = compute_transport_features(
                frame, support, loader=loader, device=device,
                with_color=bool(args.routed), **feat_kwargs)
            inputs.append(stack_fn(feats).to(torch.float16).cpu())
            gt = loader.gt(str(frame.gt_path))
            targets.append(gt.to(torch.float16).cpu())
            if i % 25 == 0:
                print(f"[train_fusion] features {i + 1}/{len(frames)}",
                      flush=True)
                torch.cuda.empty_cache()
    print(f"[train_fusion] features done in {time.time() - t0:.0f}s "
          f"({len(inputs)} views)")

    # ---- stage 2: train (random crops, fixed steps, last iterate) ----
    if args.routed:
        net = FusionNet(cin=12, out_channels=2, head_bias=(1.0, -4.0)).to(device)
    else:
        net = FusionNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=TRAIN_CONFIG["lr"],
                           weight_decay=TRAIN_CONFIG["weight_decay"])
    gen = torch.Generator().manual_seed(TRAIN_CONFIG["seed"])
    crop = int(TRAIN_CONFIG["crop"])
    losses = []
    t0 = time.time()
    for step in range(int(TRAIN_CONFIG["steps"])):
        xs, ys = [], []
        for _ in range(int(TRAIN_CONFIG["batch"])):
            vi = int(torch.randint(0, len(inputs), (1,), generator=gen))
            x = inputs[vi]
            y = targets[vi]
            h, w = x.shape[-2:]
            ch = min(crop, h)
            cw = min(crop, w)
            i0 = int(torch.randint(0, h - ch + 1, (1,), generator=gen))
            j0 = int(torch.randint(0, w - cw + 1, (1,), generator=gen))
            xs.append(x[:, i0:i0 + ch, j0:j0 + cw])
            ys.append(y[:, i0:i0 + ch, j0:j0 + cw])
        x = torch.stack(xs).to(device=device, dtype=torch.float32)
        y = torch.stack(ys).to(device=device, dtype=torch.float32)
        maps = net(x)
        base = x[:, 0:3]
        signal = x[:, 3:6]
        if args.routed:
            color = x[:, 6:9]
            # conf channel is weight_den/4 clamped; valid = conf plane > 0
            valid = x[:, 9:10] > 0.0
            pred = compose_routed(base, signal, color, valid, maps)
        else:
            pred = torch.clamp(base + maps * signal, 0.0, 1.0)
        l1 = torch.abs(pred - y).mean()
        dssim = 1.0 - ssim(pred, y)
        loss = l1 + float(TRAIN_CONFIG["dssim_weight"]) * dssim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 250 == 0:
            print(f"[train_fusion] step {step} loss {loss.item():.5f}",
                  flush=True)
    train_sec = time.time() - t0
    print(f"[train_fusion] trained {TRAIN_CONFIG['steps']} steps in "
          f"{train_sec:.0f}s; final loss {losses[-1]:.5f}")

    net_path = cache / "fusion_net.pt"
    torch.save(net.state_dict(), net_path)
    net_sha = hashlib.sha256(net_path.read_bytes()).hexdigest()
    net_mb = os.path.getsize(net_path) / (1024.0 * 1024.0)
    (cache / "fusion_training.json").write_text(json.dumps({
        "config": TRAIN_CONFIG,
        "routed": bool(args.routed),
        "inner_fuse": inner_fuse,
        "feat_kwargs": {k: v for k, v in feat_kwargs.items()},
        "n_train_views": len(frames),
        "loss_first100_mean": sum(losses[:100]) / max(len(losses[:100]), 1),
        "loss_last100_mean": sum(losses[-100:]) / max(len(losses[-100:]), 1),
        "train_seconds": train_sec,
        "net_sha256": net_sha,
        "net_mb": net_mb,
    }, indent=1) + "\n", encoding="utf-8")

    # ---- stage 3: rewrite manifest (fuse=learned|routed; sizes updated) ----
    transport["inner_fuse"] = inner_fuse
    transport["fuse"] = "routed" if args.routed else "learned"
    transport["fusion_net"] = "fusion_net.pt"
    transport["fusion_net_sha256"] = net_sha
    manifest["transport"] = transport
    sizes = manifest["sizes"]
    for rel in ("fusion_net.pt", "fusion_training.json"):
        sizes["files"][rel] = os.path.getsize(cache / rel)
    sizes["n_files"] = len(sizes["files"])
    sizes["cache_mb_raw"] = sum(sizes["files"].values()) / (1024.0 * 1024.0)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n",
                             encoding="utf-8")
    print(f"[train_fusion] wrote {net_path} ({net_mb:.1f} MB) + manifest "
          f"(fuse={transport['fuse']}, inner={inner_fuse})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
