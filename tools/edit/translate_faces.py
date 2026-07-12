#!/usr/bin/env python
"""Route-A EXP-BOUNDARY: rigid translation of a face selection — built ONLY
to bank the boundary evidence for why translation is NOT a claimed edit
class. Vertices used exclusively by selected faces are translated; shared
boundary vertices stay (contact tears are part of the demonstrated
difficulty). The decisive artifact: baked shading/shadows do not move with
the object (real scenes bake illumination into vertex colors), so the
translated object leaves its shadow behind and arrives shadowless.
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
    ap.add_argument("--cylinder", required=True)
    ap.add_argument("--offset", required=True, help="dx,dy,dz world units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-out", required=True)
    args = ap.parse_args()
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from delete_faces import fingerprint

    cyl = [float(v) for v in args.cylinder.split(",")]
    off = torch.tensor([float(v) for v in args.offset.split(",")])
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verts = sd["triangles_points"].detach()
    tri = sd["_triangle_indices"].long()
    c = torch.tensor(cyl[:3]); u = torch.tensor(cyl[3:6]); u = u / u.norm()
    rel = verts[tri].mean(1) - c
    h = rel @ u
    radial = (rel - h.unsqueeze(1) * u).norm(dim=1)
    inside = (radial < cyl[6]) & (h >= cyl[7]) & (h <= cyl[8])

    used_by_selected = torch.zeros(verts.shape[0], dtype=torch.bool)
    used_by_selected[tri[inside].reshape(-1)] = True
    used_by_rest = torch.zeros(verts.shape[0], dtype=torch.bool)
    used_by_rest[tri[~inside].reshape(-1)] = True
    exclusive = used_by_selected & ~used_by_rest

    out_sd = dict(sd)
    new_verts = verts.clone()
    new_verts[exclusive] = new_verts[exclusive] + off
    out_sd["triangles_points"] = new_verts
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_sd, args.out)
    spec = {
        "edit_type": "translate_faces_cylinder (BOUNDARY DEMO ONLY — not a "
                     "claimed edit class)",
        "cylinder": cyl, "offset": off.tolist(),
        "n_faces_selected": int(inside.sum()),
        "n_vertices_translated": int(exclusive.sum()),
        "n_shared_boundary_vertices":
            int((used_by_selected & used_by_rest).sum()),
        "parent_checkpoint": fingerprint(args.checkpoint),
        "edited_checkpoint": fingerprint(args.out),
    }
    with open(args.spec_out, "w") as fh:
        json.dump(spec, fh, indent=1)
    print(f"[translate] {spec['n_faces_selected']} faces / "
          f"{spec['n_vertices_translated']} exclusive verts moved by "
          f"{off.tolist()}; {spec['n_shared_boundary_vertices']} shared "
          f"boundary verts left (contact tears expected)")


if __name__ == "__main__":
    main()
