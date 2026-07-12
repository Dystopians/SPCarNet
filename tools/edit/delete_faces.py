#!/usr/bin/env python
"""Route A (edit-aware ECR): triangle deletion by 3D box.

Face identity = row index of `_triangle_indices`. Deletion drops the
triangle-indexed rows (indices + per-triangle stats); vertices and
per-vertex features stay (orphans are harmless). Emits the edited
checkpoint + an edit spec (json + deleted-face-id npy) with parent/edited
fingerprints for cache lineage.
"""
import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    return {"path": os.path.abspath(path),
            "sha256_first16mb": h.hexdigest(),
            "file_size_bytes": os.path.getsize(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--box", default=None,
                    help="x0,y0,z0,x1,y1,z1 (world/COLMAP units)")
    ap.add_argument("--cleanup-expand", type=float, default=None,
                    help="cylinder mode: also delete pixel_count==0 "
                         "(never-photographed) faces within radius*THIS "
                         "expanded region — unconstrained-interior cleanup")
    ap.add_argument("--extra-ids-npy", default=None,
                    help="npy of additional face ids (ORIGINAL checkpoint "
                         "indexing) to delete — union with the region "
                         "selection (e.g. probed interior-debris faces)")
    ap.add_argument("--cylinder", default=None,
                    help="cx,cy,cz,ux,uy,uz,radius,h_lo,h_hi — gravity-"
                         "aligned cylinder: axis through c along unit u; "
                         "select faces with radial dist < radius and height "
                         "along u in [h_lo, h_hi] relative to c. For scenes "
                         "whose COLMAP frame is not gravity-aligned "
                         "(protocol amendment 2026-07-12, logged).")
    ap.add_argument("--out", required=True,
                    help="edited point_cloud_state_dict.pt path")
    ap.add_argument("--spec-out", required=True, help="edit spec json path")
    args = ap.parse_args()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch

    assert (args.box is None) != (args.cylinder is None), \
        "exactly one of --box / --cylinder"
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verts = sd["triangles_points"].detach()
    tri = sd["_triangle_indices"].long()
    centroid = verts[tri].mean(dim=1)
    if args.box is not None:
        box = [float(v) for v in args.box.split(",")]
        assert len(box) == 6
        lo = torch.tensor(box[:3]); hi = torch.tensor(box[3:])
        assert bool((hi > lo).all()), "box must have positive extent"
        inside = ((centroid >= lo) & (centroid <= hi)).all(dim=1)
        region = {"kind": "box", "box": box}
    else:
        cyl = [float(v) for v in args.cylinder.split(",")]
        assert len(cyl) == 9
        c = torch.tensor(cyl[:3]); u = torch.tensor(cyl[3:6])
        u = u / u.norm()
        radius, h_lo, h_hi = cyl[6], cyl[7], cyl[8]
        rel = centroid - c
        h = rel @ u
        radial = (rel - h.unsqueeze(1) * u).norm(dim=1)
        inside = (radial < radius) & (h >= h_lo) & (h <= h_hi)
        region = {"kind": "cylinder", "center": cyl[:3], "up": u.tolist(),
                  "radius": radius, "h_lo": h_lo, "h_hi": h_hi}
    if args.cleanup_expand and args.cylinder is not None:
        # Unconstrained-interior cleanup: faces with ZERO training-view
        # visibility (checkpoint pixel_count == 0) inside an expanded copy
        # of the edit region have garbage appearance by definition (never
        # photographed) — deleting the object exposes them as debris, so
        # they are deleted WITH it. Principled: uses only banked training
        # statistics, no visual iteration.
        f = float(args.cleanup_expand)
        pad = (h_hi - h_lo) * 0.25
        exp = (radial < radius * f) & (h >= h_lo - pad) & (h <= h_hi + pad)
        unseen = sd["pixel_count"].detach() == 0
        cleanup = exp & unseen & ~inside
        inside = inside | cleanup
        region["cleanup_expand"] = f
        region["n_cleanup_unseen_faces"] = int(cleanup.sum())
    if args.extra_ids_npy:
        import numpy as _np
        extra = torch.from_numpy(_np.load(args.extra_ids_npy)).long()
        extra_mask = torch.zeros(tri.shape[0], dtype=torch.bool)
        extra_mask[extra] = True
        inside = inside | extra_mask
        region["extra_ids_npy"] = os.path.abspath(args.extra_ids_npy)
        region["n_extra_ids"] = int(extra.numel())
    deleted_ids = torch.nonzero(inside, as_tuple=False).squeeze(1)
    keep = ~inside
    n_del = int(deleted_ids.numel())
    print(f"[delete_faces] {n_del}/{tri.shape[0]} faces inside box "
          f"({100.0 * n_del / tri.shape[0]:.2f}%)")
    assert n_del > 0, "box selects no faces"

    out_sd = dict(sd)
    out_sd["_triangle_indices"] = sd["_triangle_indices"][keep]
    for key in ("importance_score", "image_size", "pixel_count"):
        if key in sd and hasattr(sd[key], "shape") \
                and sd[key].shape[0] == tri.shape[0]:
            out_sd[key] = sd[key][keep]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_sd, args.out)

    ids_npy = os.path.splitext(args.spec_out)[0] + "_deleted_ids.npy"
    np.save(ids_npy, deleted_ids.numpy().astype(np.int64))
    spec = {
        "edit_type": f"delete_faces_{region['kind']}",
        "box": region,
        "n_faces_before": int(tri.shape[0]),
        "n_faces_deleted": n_del,
        "deleted_ids_npy": os.path.abspath(ids_npy),
        "parent_checkpoint": fingerprint(args.checkpoint),
        "edited_checkpoint": fingerprint(args.out),
    }
    os.makedirs(os.path.dirname(args.spec_out), exist_ok=True)
    with open(args.spec_out, "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"[delete_faces] wrote {args.out} + {args.spec_out}")


if __name__ == "__main__":
    main()
