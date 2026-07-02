"""GEMS Stage One evaluation context (PROTOCOL.md section 4).

Loads a bare `point_cloud_state_dict.pt` checkpoint into a TriangleModel and
the scene cameras straight from the COLMAP reader (no Scene class, no model
directory needed), with training-time render settings (supersampling x4).

Purity (D4): this module consumes ONLY the checkpoint, camera poses, and eval
images. It must never import ELA / teacher / selector code.
"""
import os
import tempfile
from types import SimpleNamespace

import torch

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.triangle_model import TriangleModel
from utils.camera_utils import cameraList_from_camInfos
from triangle_renderer import render as _triangle_render
from tools.gems.scenes import SceneSpec

CKPT_FILENAME = "point_cloud_state_dict.pt"


def _camera_loader_args(spec: SceneSpec, data_device: str):
    """Minimal args namespace for utils.camera_utils.loadCam.

    loadCam reads args.resolution and args.data_device directly; every other
    attribute it touches goes through getattr(...) with a safe default.
    """
    return SimpleNamespace(
        resolution=spec.resolution,
        data_device=data_device,
        source_path=spec.source_path,
        model_path="",
        ground_masks=False,
        enable_ground_masks=False,
        ground_mask_dir="",
    )


def _read_scene_info(spec: SceneSpec):
    """Replicates the exact split rules of Scene/readColmapSceneInfo."""
    if not os.path.exists(os.path.join(spec.source_path, "sparse")):
        raise FileNotFoundError(
            f"COLMAP scene not found at {spec.source_path} (no 'sparse' dir)"
        )
    colmap = sceneLoadTypeCallbacks["Colmap"]
    if spec.split == "llff8":
        return colmap(spec.source_path, spec.images, True)
    if spec.split == "file5":
        split_file = os.path.join(spec.source_path, "split.json")
        if os.path.exists(split_file):
            return colmap(
                spec.source_path, spec.images, True,
                split_strategy="file", split_file=split_file,
            )
        return colmap(spec.source_path, spec.images, True, llffhold=5)
    raise ValueError(f"Unknown split '{spec.split}' (expected 'llff8' or 'file5')")


def _resolve_checkpoint_dir(checkpoint_path: str) -> str:
    """TriangleModel.load_parameters expects a directory that contains
    `point_cloud_state_dict.pt`; accept either that file or its directory."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        candidate = os.path.join(checkpoint_path, CKPT_FILENAME)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"{CKPT_FILENAME} not found in {checkpoint_path}")
        return checkpoint_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if os.path.basename(checkpoint_path) == CKPT_FILENAME:
        return os.path.dirname(checkpoint_path)
    # Differently-named .pt: expose it under the expected name via a symlink
    # in a temp dir so the repo loader can be reused unmodified.
    tmpdir = tempfile.mkdtemp(prefix="gems_ckpt_")
    os.symlink(checkpoint_path, os.path.join(tmpdir, CKPT_FILENAME))
    return tmpdir


class EvalContext:
    """Everything run_eval / metric modules need for one (checkpoint, scene).

    Fields: .triangles (TriangleModel on cuda, scaling=4), .pipe, .bg
    (torch [3] cuda), .train_cams (list, lazy-loaded), .test_cams (list),
    .spec (SceneSpec). Optional .out_dir is set by run_eval so metric modules
    have a durable place for per-sample arrays (npz).
    """

    def __init__(self, triangles: TriangleModel, pipe, bg: torch.Tensor,
                 spec: SceneSpec, scene_info, cam_args, checkpoint_path: str):
        self.triangles = triangles
        self.pipe = pipe
        self.bg = bg
        self.spec = spec
        self.scene_info = scene_info
        self.checkpoint_path = checkpoint_path
        self.out_dir = None
        self._cam_args = cam_args
        self._train_cams = None
        self.test_cams = cameraList_from_camInfos(scene_info.test_cameras, 1.0, cam_args)

    @property
    def train_cams(self):
        """Train cameras, loaded on first access (only geometry metrics need
        them; the pure rendering eval never pays this cost)."""
        if self._train_cams is None:
            self._train_cams = cameraList_from_camInfos(
                self.scene_info.train_cameras, 1.0, self._cam_args
            )
        return self._train_cams

    def render_view(self, cam) -> dict:
        """Forward render of one camera with training-time settings."""
        with torch.no_grad():
            return _triangle_render(cam, self.triangles, self.pipe, self.bg)

    def opaque_mask(self) -> torch.Tensor:
        """Bool [T], all True: PROTOCOL 1.1.0 §4.3 opaque surface = ALL
        checkpoint triangles.

        Rationale (changelog 1.1.0): TriangleModel pins render-time opacity to
        >= 0.999 for EVERY triangle (opacity floor,
        scene/triangle_model.py:347 `self.opacity_floor = 0.999`), so the
        renderer draws all triangles near-opaque and there is no meaningful
        translucency in this representation (render-time opacity =
        floor + (1 - floor) * sigmoid(weight) is in [0.999, 1.0] for any
        weight; the weight logits themselves are frozen, lr = 0.0). The
        pre-1.1.0 rule (min vertex sigmoid(weight) >= 0.5) thresholded the
        raw logits and did NOT describe the rendered surface — on garden it
        selected only ~0.26% of triangles. g4/d1/d2 therefore operate on all
        faces.

        The method is retained (metric modules duck-type against it) but is
        now definitionally all-True; callers may equivalently use all faces.
        """
        faces = self.faces()
        return torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)

    def vertices(self) -> torch.Tensor:
        return self.triangles.vertices.detach()

    def faces(self) -> torch.Tensor:
        return self.triangles._triangle_indices.detach().long()

    def finite_faces_mask(self) -> torch.Tensor:
        """Bool [T]: faces whose 3 vertices are all finite (PROTOCOL §4.3).

        Training can rarely produce NaN vertices (observed: 2/1.97M verts on
        toy_parking clean30k -> 13 NaN faces); the rasterizer silently culls
        them, so they are not part of the rendered surface. Geometry and
        downstream surfaces exclude them and report the count. Indexing is
        preserved: this is a mask over the ORIGINAL face order (rend_ids
        compatibility).
        """
        finite_v = torch.isfinite(self.vertices()).all(dim=1)
        return finite_v[self.faces()].all(dim=1)


def build_eval_context(checkpoint_path: str, spec: SceneSpec,
                       data_device: str = "cpu", sh_degree: int = 3,
                       supersampling: int = 4) -> EvalContext:
    """Load checkpoint + cameras and return a ready EvalContext.

    data_device='cpu' keeps GT images in host RAM (moved to GPU per view);
    render math always runs on cuda, matching render.py numerics.
    """
    if not spec.exists:
        raise FileNotFoundError(
            f"scene '{spec.name}' is registered but its dataset does not exist yet"
        )
    scene_info = _read_scene_info(spec)
    cam_args = _camera_loader_args(spec, data_device)

    triangles = TriangleModel(sh_degree)
    triangles.scaling = supersampling  # render.py training-time supersampling
    triangles.load_parameters(_resolve_checkpoint_dir(checkpoint_path), device="cuda")

    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=1.0,
        debug=False,
    )
    bg_color = [1.0, 1.0, 1.0] if spec.white_background else [0.0, 0.0, 0.0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return EvalContext(triangles, pipe, bg, spec, scene_info, cam_args,
                       os.path.abspath(checkpoint_path))
