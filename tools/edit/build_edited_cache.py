#!/usr/bin/env python
"""Route A (edit-aware ECR): build the LOCALLY-INVALIDATED edited cache (C5).

Two passes:
  1. ORIGINAL checkpoint: render each train view's per-pixel face ids
     (`rend_ids`) -> stale mask = isin(ids, deleted set), 1-px dilated;
     affected view <=> any stale pixel.
  2. EDITED checkpoint: regenerate renders + median depths for AFFECTED
     views only; hardlink everything else (locality = bytes actually
     written).
GT photographs are hardlinked UNMASKED on disk — invalidation happens at
the single warp site via the manifest's edit.masks entries (masks can only
remove evidence). Transport config, calibrated (K, alpha) and the frozen
fusion net are REUSED from the original cache (frozen protocol decision:
isolates the invalidation mechanism).
"""
import argparse
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-cache", required=True)
    ap.add_argument("--edited-checkpoint", required=True)
    ap.add_argument("--spec", required=True, help="edit spec json")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparse", action="store_true",
                    help="sparse sidecar: store only changed-region patches "
                         "for affected views (renders/depths hardlink the "
                         "ORIGINAL; EcrRenderer composes at load)")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from tools.gems.scenes import SCENES
    from tools.gems.eval_context import build_eval_context
    from run_eval import checkpoint_fingerprint

    spec = json.load(open(args.spec))
    deleted = torch.from_numpy(np.load(spec["deleted_ids_npy"])).cuda()
    src = args.original_cache
    dst = args.out
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    train_views = list(manifest["train_views"])
    orig_ckpt = spec["parent_checkpoint"]["path"]
    assert checkpoint_fingerprint(orig_ckpt)["sha256_first16mb"] == \
        manifest["checkpoint"]["sha256_first16mb"], \
        "original cache does not belong to the spec's parent checkpoint"

    for d in ("renders", "gt", "depths", "masks"):
        os.makedirs(os.path.join(dst, d), exist_ok=True)

    # ---- pass 1: ORIGINAL checkpoint -> stale masks + affected set ----
    t0 = time.time()
    ctx = build_eval_context(orig_ckpt, SCENES[args.scene])
    train_cams = {str(c.image_name): c for c in ctx.train_cams}
    # chained edits (red-team fix 2026-07-12): a parent cache may itself be
    # edited — its masks MUST be inherited (valid = parent_valid AND new_valid)
    # or a second edit would silently re-validate the first edit's stale
    # evidence. Parent affected views stay affected.
    parent_masks = dict(manifest.get("edit", {}).get("masks", {}))
    from utils.evidence_lumigraph_adapter import read_image_tensor
    from pathlib import Path as _P
    affected, masks_rel = [], {}
    stale_px_total = 0
    with torch.no_grad():
        for name in train_views:
            cam = train_cams[name]
            pkg = ctx.render_view(cam)
            ids = pkg["rend_ids"].detach().float()
            if ids.ndim == 3:
                ids = ids.unsqueeze(0)
            h, w = int(cam.image_height), int(cam.image_width)
            ids = F.interpolate(ids, size=(h, w), mode="nearest") \
                .squeeze().long().cuda()
            stale = torch.isin(ids, deleted)
            parent_valid = None
            if name in parent_masks:
                parent_valid = read_image_tensor(
                    _P(src) / parent_masks[name], device="cuda")[0]
            if not bool(stale.any()) and parent_valid is None:
                continue
            # 1-px dilation of the STALE region (splat bleed)
            stale_f = stale.float()[None, None]
            stale_d = F.max_pool2d(stale_f, 3, stride=1, padding=1) \
                .squeeze() > 0.5
            valid = (~stale_d).float()
            if parent_valid is not None:
                valid = torch.minimum(valid, parent_valid)
            rel = f"masks/{name}.png"
            torchvision.utils.save_image(valid[None],
                                         os.path.join(dst, rel))
            masks_rel[name] = rel
            affected.append(name)
            stale_px_total += int((valid < 0.5).sum())
    del ctx
    torch.cuda.empty_cache()
    t_mask = time.time() - t0
    print(f"[edited_cache] affected {len(affected)}/{len(train_views)} views,"
          f" {stale_px_total} stale px, mask pass {t_mask:.0f}s", flush=True)

    # ---- pass 2: EDITED checkpoint -> renders/depths for affected only ----
    t0 = time.time()
    ctx = build_eval_context(args.edited_checkpoint, SCENES[args.scene])
    train_cams = {str(c.image_name): c for c in ctx.train_cams}
    ext = manifest.get("image_ext", {})
    rext = str(ext.get("renders", "png"))
    gext = str(ext.get("gt", "png"))
    bytes_written = 0
    patches_rel = {}
    if args.sparse:
        os.makedirs(os.path.join(dst, "patches"), exist_ok=True)
    with torch.no_grad():
        for name in train_views:
            rdst = os.path.join(dst, "renders", f"{name}.{rext}")
            ddst = os.path.join(dst, "depths", f"{name}.npy")
            if name in masks_rel and args.sparse:
                pkg = ctx.render_view(train_cams[name])
                new_r = pkg["render"][:3].clamp(0, 1)
                new_d = pkg["surf_depth"][0]
                from utils.evidence_lumigraph_adapter import (
                    read_image_tensor, read_depth_tensor)
                from pathlib import Path as __P
                old_r = read_image_tensor(
                    __P(src) / "renders" / f"{name}.{rext}", device="cuda")
                old_d = read_depth_tensor(
                    __P(src) / "depths" / f"{name}.npy", device="cuda")
                # bit-identity contract: a pixel is "changed" iff its
                # QUANTIZED (PNG-serialized) value or exact f32 depth differs
                q = lambda x: x.mul(255).add(0.5).clamp(0, 255).to(torch.uint8)
                changed = (q(new_r) != q(old_r)).any(0)
                changed |= new_d != old_d
                ys, xs = torch.nonzero(changed, as_tuple=True)
                if ys.numel() == 0:
                    y0 = x0 = 0; y1 = x1 = 1
                else:
                    pad = 4
                    y0 = max(int(ys.min()) - pad, 0)
                    x0 = max(int(xs.min()) - pad, 0)
                    y1 = min(int(ys.max()) + pad + 1, changed.shape[0])
                    x1 = min(int(xs.max()) + pad + 1, changed.shape[1])
                prel = f"patches/{name}.npz"
                r8 = (new_r[:, y0:y1, x0:x1].permute(1, 2, 0) * 255.0) \
                    .add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                np.savez_compressed(
                    os.path.join(dst, prel),
                    bbox=np.array([y0, x0, y1, x1], dtype=np.int64),
                    render=r8,
                    depth=new_d[y0:y1, x0:x1].detach().cpu().numpy()
                    .astype(np.float32))
                patches_rel[name] = prel
                bytes_written += os.path.getsize(os.path.join(dst, prel))
                for s, d in ((os.path.join(src, "renders", f"{name}.{rext}"),
                              rdst),
                             (os.path.join(src, "depths", f"{name}.npy"),
                              ddst)):
                    if not os.path.exists(d):
                        os.link(s, d)
            elif name in masks_rel:
                pkg = ctx.render_view(train_cams[name])
                torchvision.utils.save_image(
                    pkg["render"][:3].clamp(0, 1), rdst)
                np.save(ddst, pkg["surf_depth"][0].detach().cpu().numpy()
                        .astype(np.float32))
                bytes_written += os.path.getsize(rdst) + os.path.getsize(ddst)
            else:
                for s, d in ((os.path.join(src, "renders", f"{name}.{rext}"),
                              rdst),
                             (os.path.join(src, "depths", f"{name}.npy"),
                              ddst)):
                    if not os.path.exists(d):
                        os.link(s, d)
            gdst = os.path.join(dst, "gt", f"{name}.{gext}")
            if not os.path.exists(gdst):
                os.link(os.path.join(src, "gt", f"{name}.{gext}"), gdst)
    del ctx
    torch.cuda.empty_cache()
    t_rebuild = time.time() - t0
    mask_bytes = sum(os.path.getsize(os.path.join(dst, m))
                     for m in masks_rel.values())

    # ---- manifest: lineage + reuse of transport/alpha/net ----
    shutil.copyfile(os.path.join(src, "camera_index.json"),
                    os.path.join(dst, "camera_index.json"))
    for extra in ("fusion_net.pt", "fusion_training.json"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, extra))
    out_man = dict(manifest)
    out_man["checkpoint"] = checkpoint_fingerprint(args.edited_checkpoint)
    out_man["edit"] = {
        "type": spec["edit_type"],
        "box": spec["box"],
        "n_faces_deleted": spec["n_faces_deleted"],
        "parent_checkpoint": spec["parent_checkpoint"],
        "spec_path": os.path.abspath(args.spec),
        "affected_views": affected,
        "masks": masks_rel,
        **({"patches": patches_rel} if patches_rel else {}),
        "policy": "evidence masked at warp (multiplicative <=1); "
                  "renders/depths regenerated for affected views only; "
                  "(K, alpha) + fusion net reused from parent cache",
    }
    files = {}
    for root, _, names in os.walk(dst):
        for n in names:
            p = os.path.join(root, n)
            files[os.path.relpath(p, dst)] = os.path.getsize(p)
    out_man["sizes"] = {
        "files": files, "n_files": len(files),
        "cache_mb_raw": sum(files.values()) / (1024.0 * 1024.0),
        "cache_mb_compressed": manifest.get("sizes", {}).get(
            "cache_mb_compressed", -1.0),
    }
    out_man["update_cost"] = {
        "mask_pass_seconds": t_mask,
        "local_rebuild_seconds": t_rebuild,
        "bytes_rewritten_renders_depths": bytes_written,
        "bytes_masks": mask_bytes,
        "n_affected_views": len(affected),
        "n_train_views": len(train_views),
    }
    with open(os.path.join(dst, "manifest.json"), "w") as fh:
        json.dump(out_man, fh, indent=1)
    print(f"[edited_cache] wrote {dst}: {len(affected)} affected, "
          f"{(bytes_written + mask_bytes) / 1e6:.1f} MB rewritten "
          f"({t_mask + t_rebuild:.0f}s total)")


if __name__ == "__main__":
    main()
