#!/usr/bin/env python
"""Route A class-2 (appearance-only edit): recolor the vertices of a face
selection by a hue/value transform on features_dc (SH DC term). Features are
per-VERTEX: the transform applies to every vertex used by a selected face
(shared boundary vertices shift too — reported, not hidden). Geometry is
untouched, so stale evidence remains DEPTH-CONSISTENT — the class where the
z-test gives no accidental protection and masking is the only defense.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cylinder", required=True,
                    help="cx,cy,cz,ux,uy,uz,radius,h_lo,h_hi")
    ap.add_argument("--rgb-shift", required=True,
                    help="dr,dg,db added to the DC color (activated-space "
                         "approx: DC is linear in color)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-out", required=True)
    args = ap.parse_args()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import numpy as np
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from delete_faces import fingerprint

    cyl = [float(v) for v in args.cylinder.split(",")]
    shift = torch.tensor([float(v) for v in args.rgb_shift.split(",")])
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verts = sd["triangles_points"].detach()
    tri = sd["_triangle_indices"].long()
    c = torch.tensor(cyl[:3]); u = torch.tensor(cyl[3:6]); u = u / u.norm()
    rel = verts[tri].mean(1) - c
    h = rel @ u
    radial = (rel - h.unsqueeze(1) * u).norm(dim=1)
    inside = (radial < cyl[6]) & (h >= cyl[7]) & (h <= cyl[8])
    face_ids = torch.nonzero(inside, as_tuple=False).squeeze(1)
    vert_ids = torch.unique(tri[inside].reshape(-1))
    out_sd = dict(sd)
    fdc = sd["features_dc"].detach().clone()
    fdc[vert_ids, 0, :] = fdc[vert_ids, 0, :] + shift
    out_sd["features_dc"] = fdc
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_sd, args.out)
    ids_npy = os.path.splitext(args.spec_out)[0] + "_deleted_ids.npy"
    np.save(ids_npy, face_ids.numpy().astype(np.int64))
    spec = {
        "edit_type": "recolor_faces_cylinder",
        "box": {"kind": "cylinder", "center": cyl[:3], "up": u.tolist(),
                "radius": cyl[6], "h_lo": cyl[7], "h_hi": cyl[8]},
        "rgb_shift": shift.tolist(),
        "n_faces_before": int(tri.shape[0]),
        "n_faces_deleted": int(face_ids.numel()),  # = affected faces
        "n_vertices_shifted": int(vert_ids.numel()),
        "deleted_ids_npy": os.path.abspath(ids_npy),
        "parent_checkpoint": fingerprint(args.checkpoint),
        "edited_checkpoint": fingerprint(args.out),
    }
    with open(args.spec_out, "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"[recolor] {int(face_ids.numel())} faces / "
          f"{int(vert_ids.numel())} vertices shifted by {shift.tolist()}; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
