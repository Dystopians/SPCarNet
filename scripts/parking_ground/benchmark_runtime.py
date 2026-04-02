#!/usr/bin/env python3
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel, render
from utils.general_utils import safe_state


def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = ArgumentParser(description="Controlled runtime benchmark without PNG I/O.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--warmup_views", default=5, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--max_views", default=0, type=int, help="0 means all test views.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(
            args=dataset,
            triangles=triangles,
            init_opacity=None,
            set_sigma=None,
            load_iteration=args.iteration,
            shuffle=False,
        )
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        views = list(scene.getTestCameras())
        if int(args.max_views) > 0:
            views = views[: int(args.max_views)]

        if len(views) == 0:
            raise RuntimeError("No test views available for runtime benchmark.")

        warmup_views = views[: min(len(views), max(0, int(args.warmup_views)))]
        for view in warmup_views:
            pkg = render(view, triangles, pipe, background)
            _ = pkg["render"]
        _sync_if_needed()

        per_repeat_fps = []
        per_repeat_elapsed = []
        for _rep in range(max(1, int(args.repeats))):
            _sync_if_needed()
            t0 = time.perf_counter()
            for view in views:
                pkg = render(view, triangles, pipe, background)
                _ = pkg["render"]
            _sync_if_needed()
            elapsed = max(1e-9, time.perf_counter() - t0)
            per_repeat_elapsed.append(float(elapsed))
            per_repeat_fps.append(float(len(views)) / float(elapsed))

        result = {
            "model_path": str(dataset.model_path),
            "loaded_iteration": int(scene.loaded_iter),
            "num_views": int(len(views)),
            "warmup_views": int(len(warmup_views)),
            "repeats": int(max(1, int(args.repeats))),
            "runtime_fps_mean": float(np.mean(per_repeat_fps)),
            "runtime_fps_std": float(np.std(per_repeat_fps)),
            "repeat_elapsed_sec": [float(v) for v in per_repeat_elapsed],
            "repeat_fps": [float(v) for v in per_repeat_fps],
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
