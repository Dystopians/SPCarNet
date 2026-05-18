#!/usr/bin/env python3
"""Fit train-certified face-local residual appearance deltas.

This operator is a representation-level successor to the shared-vertex SH
delta.  Instead of changing the SH coefficients of vertices shared by many
faces, it duplicates the three vertices of train-certified high-residual faces
and redirects only those faces to the local copies.  Geometry and triangle count
are preserved; the added local vertices carry a bounded SH residual state.

No held-out test residuals are read.  Fitting uses train-cache views and a
deterministic train policy-validation split decides which face-local deltas are
materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import copy_model_metadata, checkpoint_path, validate_faces
from utils.sh_utils import C0, C1, C2, C3


@dataclass
class PixelSamples:
    face_ids: np.ndarray
    barycentric: np.ndarray
    residual_rgb: np.ndarray
    weights: np.ndarray
    camera_centers: np.ndarray
    view_names: list[str]
    region_bins: np.ndarray

    @property
    def count(self) -> int:
        return int(self.face_ids.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.80)
    parser.add_argument("--min_pixel_count", type=float, default=6.0)
    parser.add_argument("--max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--max_total_samples", type=int, default=320000)
    parser.add_argument("--high_error_quantile", type=float, default=0.65)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument(
        "--face_score_weight_power",
        type=float,
        default=0.0,
        help=(
            "Optional train-only saliency weighting for sampled residual fitting. "
            "When >0, per-pixel residual weights are multiplied by "
            "(face_score / median_selected_face_score) ** power and clipped by "
            "--face_score_weight_max. Default 0 preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--face_score_weight_max",
        type=float,
        default=4.0,
        help="Upper clip for --face_score_weight_power saliency weights.",
    )
    parser.add_argument(
        "--region_carrier_json",
        type=Path,
        default=None,
        help=(
            "Optional render-visible region carrier JSON. When provided, sampled "
            "train residuals are reweighted by per-view region core/context support. "
            "No held-out test views are read."
        ),
    )
    parser.add_argument("--region_core_weight", type=float, default=1.0)
    parser.add_argument("--region_context_weight", type=float, default=1.0)
    parser.add_argument("--region_outside_weight", type=float, default=1.0)
    parser.add_argument("--region_boundary_px", type=int, default=0)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.35)
    parser.add_argument(
        "--uniform_barycentric",
        action="store_true",
        help="Use equal 1/3 weights when the evidence cache does not contain barycentric maps.",
    )
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.18)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.014)
    parser.add_argument(
        "--max_abs_sh_coeff",
        type=float,
        default=0.0,
        help="Bound for each non-DC SH coefficient delta. 0 derives it from max_abs_delta_rgb / C1.",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Face-local residual SH degree. 1 preserves historical behavior; 3 uses the full stored SH basis.",
    )
    parser.add_argument("--lambda_mag", type=float, default=2e-2)
    parser.add_argument("--lambda_sh1_mag", type=float, default=5e-2)
    parser.add_argument("--lambda_smooth", type=float, default=8e-2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument(
        "--shared_residual_field",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit one train-only RBF residual field shared by all selected face-local "
            "corner slots instead of independent per-face local coefficients. The "
            "field is still baked into duplicated face-local vertices, so existing "
            "render/gate/portfolio code can evaluate it without renderer changes."
        ),
    )
    parser.add_argument("--shared_residual_field_anchors", type=int, default=16)
    parser.add_argument(
        "--shared_residual_field_sigma",
        type=float,
        default=0.0,
        help="RBF sigma in normalized scene units. 0 chooses a deterministic anchor-distance median.",
    )
    parser.add_argument("--shared_residual_field_lr", type=float, default=0.0)
    parser.add_argument("--shared_residual_field_weight_l2", type=float, default=1.0e-4)
    parser.add_argument(
        "--shared_residual_field_view_hinge_weight",
        type=float,
        default=0.0,
        help=(
            "Train-only per-view safety hinge. Penalizes fit-train views whose "
            "weighted residual MSE would increase under the shared field."
        ),
    )
    parser.add_argument("--shared_residual_field_view_hinge_min_samples", type=int, default=16)
    parser.add_argument(
        "--shared_residual_field_duplicate_smooth_weight",
        type=float,
        default=0.0,
        help=(
            "Penalize coefficient disagreement among duplicated local slots that "
            "originate from the same source mesh vertex."
        ),
    )
    parser.add_argument("--max_faces_to_apply", type=int, default=2048)
    parser.add_argument("--min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--min_policy_val_samples", type=int, default=512)
    parser.add_argument("--min_policy_val_unique_faces", type=int, default=16)
    parser.add_argument(
        "--validation_shrink_mode",
        choices=("none", "global", "face", "global_gain", "face_gain"),
        default="none",
        help=(
            "Train-only residual amplitude calibration. 'global' fits one shrink "
            "scale on policy-val samples; 'face' fits one shrink scale per selected face. "
            "'global_gain'/'face_gain' use the same policy-val-only closed-form scale but "
            "allow bounded amplification up to --validation_gain_max_scale."
        ),
    )
    parser.add_argument(
        "--validation_gain_max_scale",
        type=float,
        default=1.0,
        help="Maximum train-only amplitude scale for validation_shrink_mode '*_gain'.",
    )
    parser.add_argument(
        "--validation_shrink_min_samples",
        type=int,
        default=8,
        help="Minimum policy-val samples required before a face gets a nonzero face shrink scale.",
    )
    parser.add_argument(
        "--crossfold_gain_certificate_folds",
        type=int,
        default=0,
        help=(
            "If >1, split train evidence views into this many interleaved folds "
            "and require each accepted face to have nonnegative proxy gain across enough folds. "
            "This is an all-train fold-consistency check, not an independent cross-fit certificate."
        ),
    )
    parser.add_argument("--crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument("--min_face_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--min_face_policy_val_samples", type=int, default=8)
    parser.add_argument(
        "--min_face_gain_certificate_views",
        type=int,
        default=0,
        help=(
            "If >0, require each accepted face to have predicted residual MSE gain "
            "on at least this many policy-val train views."
        ),
    )
    parser.add_argument(
        "--min_face_gain_certificate_relative_gain",
        type=float,
        default=0.0,
        help="Minimum per-view relative MSE gain for one policy-val train view to certify a face.",
    )
    parser.add_argument(
        "--min_face_gain_certificate_view_samples",
        type=int,
        default=4,
        help="Minimum samples from one policy-val train view before it can certify a face.",
    )
    parser.add_argument(
        "--min_face_gain_certificate_fraction",
        type=float,
        default=0.0,
        help="Optional minimum fraction of eligible policy-val train views that must certify a face.",
    )
    parser.add_argument(
        "--min_face_view_consensus",
        type=float,
        default=0.0,
        help=(
            "If >0, require this fraction of policy-val train views for a face "
            "to agree with the face residual direction before materializing its local vertices."
        ),
    )
    parser.add_argument(
        "--min_face_consensus_views",
        type=int,
        default=2,
        help="Minimum policy-val train views needed for the face/view consensus certificate.",
    )
    parser.add_argument(
        "--min_face_consensus_view_samples",
        type=int,
        default=4,
        help="Minimum samples from one policy-val train view before it votes in face/view consensus.",
    )
    parser.add_argument(
        "--face_consensus_min_cosine",
        type=float,
        default=0.0,
        help="Minimum cosine against the per-face residual direction for one view to count as agreeing.",
    )
    parser.add_argument(
        "--patch_cert_rings",
        type=int,
        default=0,
        help=(
            "If >0, grow accepted face seeds into connected train-evidence patches "
            "using selected-face adjacency and require a patch-level train policy-val gain."
        ),
    )
    parser.add_argument("--patch_cert_max_faces_per_seed", type=int, default=8)
    parser.add_argument("--patch_cert_min_direction_cosine", type=float, default=0.90)
    parser.add_argument("--patch_cert_min_neighbor_policy_val_samples", type=int, default=4)
    parser.add_argument("--patch_cert_min_neighbor_policy_val_relative_gain", type=float, default=-0.02)
    parser.add_argument("--patch_cert_min_policy_val_samples", type=int, default=16)
    parser.add_argument("--patch_cert_min_relative_gain", type=float, default=0.0)
    parser.add_argument("--patch_cert_neighbor_mode", choices=("topology", "centroid", "both"), default="topology")
    parser.add_argument("--patch_cert_centroid_candidates_per_seed", type=int, default=64)
    parser.add_argument(
        "--patch_cert_seed_rescue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When strict single-face certificates leave too few seeds, propose extra seeds "
            "from the train policy-val face pool using a fixed group-first rule. The rescued "
            "seeds still have to pass patch-level fold certificates, carrier-basis checks, "
            "post-shrink policy validation, and the outer train-val gate."
        ),
    )
    parser.add_argument(
        "--patch_cert_seed_rescue_min_candidates",
        type=int,
        default=1,
        help="Trigger seed rescue only when strict face candidates are below this count.",
    )
    parser.add_argument(
        "--patch_cert_seed_rescue_max_seeds",
        type=int,
        default=16,
        help="Maximum number of deterministic group-first rescue seeds to append.",
    )
    parser.add_argument(
        "--patch_cert_seed_rescue_min_aux_witnesses",
        type=int,
        default=1,
        help=(
            "Minimum number of enabled auxiliary face witnesses that must pass for a rescue "
            "seed. Auxiliary witnesses are face-view gain, face crossfold gain, and "
            "face-view residual consensus."
        ),
    )
    parser.add_argument(
        "--patch_cert_crossfold_folds",
        type=int,
        default=0,
        help=(
            "If >1, require each accepted patch carrier to pass a train-only fold proxy-gain "
            "certificate. This gates the patch itself, not only the seed face."
        ),
    )
    parser.add_argument("--patch_cert_crossfold_min_passing_folds", type=int, default=0)
    parser.add_argument("--patch_cert_crossfold_min_fold_relative_gain", type=float, default=0.0)
    parser.add_argument("--patch_cert_crossfold_min_fold_samples", type=int, default=4)
    parser.add_argument(
        "--patch_cert_neighbor_crossfold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When patch-fold certification is enabled, require each neighbor to pass "
            "the same train-only fold certificate before it can enter a patch."
        ),
    )
    parser.add_argument(
        "--patch_cert_shrink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When patch certification is enabled, fit one train-only shrink scale per accepted patch.",
    )
    parser.add_argument(
        "--patch_cert_cluster_basis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit one shared corner-slot residual-SH basis per accepted PatchCert carrier before "
            "shrink/crossfold checks, replacing independent face-row coefficients inside that carrier."
        ),
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_mode",
        choices=("shared", "scaled", "rank2", "chart_linear", "chart_quad", "field_linear", "field_quad"),
        default="shared",
        help=(
            "Carrier-basis parameterization: 'shared' copies one corner-slot basis to every face; "
            "'scaled' shares the basis but learns one positive face scale per carrier face; "
            "'rank2' learns two shared corner-slot bases with per-face nonnegative simplex weights; "
            "'chart_linear' fits a constant+linear residual field over a local patch chart; "
            "'chart_quad' adds bounded quadratic chart terms; 'field_linear'/'field_quad' add "
            "train-view hinge consistency and source-vertex field-continuity regularization."
        ),
    )
    parser.add_argument("--patch_cert_cluster_basis_steps", type=int, default=240)
    parser.add_argument("--patch_cert_cluster_basis_lr", type=float, default=0.025)
    parser.add_argument("--patch_cert_cluster_basis_min_samples", type=int, default=32)
    parser.add_argument(
        "--patch_cert_cluster_basis_max_scale",
        type=float,
        default=2.0,
        help="Maximum per-face positive scale for --patch_cert_cluster_basis_mode scaled.",
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_max_fit_mse_regression",
        type=float,
        default=0.02,
        help=(
            "Reject a shared-basis patch if its train-fit MSE exceeds the independent face-local "
            "fit by more than this relative amount."
        ),
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_init",
        choices=("mean", "zero"),
        default="mean",
        help="Initialize the shared patch basis from the mean face-local coefficients or from zero.",
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_view_hinge_weight",
        type=float,
        default=0.0,
        help=(
            "For field_* carrier modes, penalize train views whose patch residual field "
            "increases weighted residual MSE. This is a train-only view-consistency term."
        ),
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_view_hinge_min_samples",
        type=int,
        default=16,
        help="Minimum samples in one train view before it contributes to the field_* view hinge loss.",
    )
    parser.add_argument(
        "--patch_cert_cluster_basis_geometry_smooth_weight",
        type=float,
        default=0.0,
        help=(
            "For field_* carrier modes, penalize discontinuities between field coefficients "
            "at source vertices shared by carrier faces before local vertex duplication."
        ),
    )
    parser.add_argument(
        "--strict_patchcert_carrier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the paper-facing strict PatchCert carrier policy: patch growth, "
            "patch-fold certification, neighbor fold admission, post-shrink checks, "
            "and certified whole-carrier plan replay."
        ),
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_selector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After PatchCert growth, select whole carriers with a deterministic "
            "train-only view-holdout stability rule. This never splits a carrier "
            "and does not read held-out test views."
        ),
    )
    parser.add_argument("--patch_cert_carrier_holdout_groups", type=int, default=4)
    parser.add_argument(
        "--patch_cert_carrier_holdout_grouping",
        choices=("view", "sample_balanced"),
        default="view",
        help=(
            "How to form policy-val carrier holdout groups. 'view' keeps disjoint "
            "policy-val view groups. 'sample_balanced' deterministically splits "
            "policy-val samples, which is useful when a carrier is visible in too "
            "few policy-val views for view-level voting."
        ),
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_disjoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reserve a deterministic half of the policy-val train samples for carrier "
            "holdout and use the other half for policy tuning/shrink. This keeps the "
            "carrier certificate train-only but disjoint from validation shrink samples."
        ),
    )
    parser.add_argument("--patch_cert_carrier_holdout_min_passing_groups", type=int, default=3)
    parser.add_argument("--patch_cert_carrier_holdout_min_group_relative_gain", type=float, default=0.0)
    parser.add_argument("--patch_cert_carrier_holdout_min_group_samples", type=int, default=4)
    parser.add_argument(
        "--patch_cert_carrier_holdout_max_mse_regression",
        type=float,
        default=0.0,
        help="Maximum per-holdout-group relative MSE regression allowed for a carrier.",
    )
    parser.add_argument("--patch_cert_carrier_holdout_cvar_fraction", type=float, default=0.25)
    parser.add_argument("--patch_cert_carrier_holdout_cvar_weight", type=float, default=1.0)
    parser.add_argument(
        "--patch_cert_carrier_holdout_max_carriers",
        type=int,
        default=0,
        help="Optional fixed cap on selected carriers after holdout ranking. 0 keeps all passing carriers.",
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_auto_prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Select a deterministic train-only prefix of holdout-stable carriers by "
            "maximizing cumulative carrier-holdout score after score sorting. This "
            "turns top-k carrier sweeps into a fixed auditable policy."
        ),
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_auto_prefix_min_faces",
        type=int,
        default=0,
        help=(
            "Optional train-only coverage floor for auto-prefix selection. When >0, "
            "a cumulative carrier prefix must materialize at least this many unique "
            "faces before it can be selected; otherwise the operator falls back to "
            "no-op instead of reporting a numerically tiny accepted repair."
        ),
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_auto_prefix_face_bonus",
        type=float,
        default=0.0,
        help=(
            "Optional coverage bonus added to the auto-prefix ranking key as "
            "bonus * log1p(prefix_faces). Defaults to 0 for backward-compatible "
            "score-only prefix selection."
        ),
    )
    parser.add_argument(
        "--patch_cert_carrier_holdout_auto_prefix_positive_tail_safe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When auto-prefix is enabled, stop before the first passed carrier whose "
            "individual holdout certificate is not positive and tail-safe. This may "
            "select a safe prefix below --patch_cert_carrier_holdout_auto_prefix_min_faces "
            "instead of adding a negative/tail carrier only to satisfy coverage."
        ),
    )
    parser.add_argument(
        "--candidate_plan_out",
        type=Path,
        default=None,
        help=(
            "Write the final train-certified accepted face-local residual carrier and fitted "
            "coefficients to a JSON plan for later materialization."
        ),
    )
    parser.add_argument(
        "--materialize_plan_in",
        type=Path,
        default=None,
        help="Materialize face-local residuals from a previously written candidate plan instead of refitting.",
    )
    parser.add_argument(
        "--materialize_plan_limit",
        type=int,
        default=0,
        help="Keep only the first N rows from --materialize_plan_in after optional face-id filtering. 0 keeps all.",
    )
    parser.add_argument(
        "--materialize_plan_face_ids",
        default="",
        help="Optional comma-separated face ids to materialize from --materialize_plan_in.",
    )
    parser.add_argument(
        "--materialize_plan_scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to plan coefficients during materialization. Used only with --materialize_plan_in.",
    )
    parser.add_argument(
        "--materialize_plan_alpha_json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing per-face alpha multipliers for materialized plan rows. "
            "Supported forms: {'face_alphas': {'123': 0.5}} or "
            "{'face_alphas': [{'face_id': 123, 'alpha': 0.5}]}."
        ),
    )
    parser.add_argument(
        "--materialize_plan_render_trust_json",
        type=Path,
        default=None,
        help=(
            "Optional train-val render-space certificate authorizing a non-unit "
            "--materialize_plan_scale under strict plan replay. The certificate "
            "must be accepted, test-free, match the requested scale, and match "
            "the plan sha256 when provided."
        ),
    )
    parser.add_argument(
        "--materialize_allow_uncertified_plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow materializing legacy plan rows that do not carry explicit "
            "policy/PatchCert certification. Default is strict: uncertified "
            "rows are rejected so plan replay cannot bypass the train-only gate."
        ),
    )
    parser.add_argument("--no_op_on_fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_apply", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not math.isfinite(float(args.face_score_weight_power)) or float(args.face_score_weight_power) < 0.0:
        parser.error("--face_score_weight_power must be finite and >= 0")
    if not math.isfinite(float(args.face_score_weight_max)) or float(args.face_score_weight_max) < 1.0:
        parser.error("--face_score_weight_max must be finite and >= 1")
    for name in ("region_core_weight", "region_context_weight", "region_outside_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and >= 0")
    if int(args.region_boundary_px) < 0:
        parser.error("--region_boundary_px must be >= 0")
    if float(args.patch_cert_cluster_basis_max_scale) <= 0.0:
        parser.error("--patch_cert_cluster_basis_max_scale must be > 0")
    if float(args.patch_cert_cluster_basis_lr) <= 0.0:
        parser.error("--patch_cert_cluster_basis_lr must be > 0")
    if int(args.patch_cert_cluster_basis_steps) < 0:
        parser.error("--patch_cert_cluster_basis_steps must be >= 0")
    if int(args.patch_cert_cluster_basis_min_samples) <= 0:
        parser.error("--patch_cert_cluster_basis_min_samples must be > 0")
    if int(args.patch_cert_seed_rescue_min_candidates) < 0:
        parser.error("--patch_cert_seed_rescue_min_candidates must be >= 0")
    if int(args.patch_cert_seed_rescue_max_seeds) < 0:
        parser.error("--patch_cert_seed_rescue_max_seeds must be >= 0")
    if int(args.patch_cert_seed_rescue_min_aux_witnesses) < 0:
        parser.error("--patch_cert_seed_rescue_min_aux_witnesses must be >= 0")
    if int(args.shared_residual_field_anchors) <= 0:
        parser.error("--shared_residual_field_anchors must be > 0")
    if not math.isfinite(float(args.shared_residual_field_sigma)) or float(args.shared_residual_field_sigma) < 0.0:
        parser.error("--shared_residual_field_sigma must be finite and >= 0")
    if not math.isfinite(float(args.shared_residual_field_lr)) or float(args.shared_residual_field_lr) < 0.0:
        parser.error("--shared_residual_field_lr must be finite and >= 0")
    if not math.isfinite(float(args.shared_residual_field_weight_l2)) or float(args.shared_residual_field_weight_l2) < 0.0:
        parser.error("--shared_residual_field_weight_l2 must be finite and >= 0")
    if (
        not math.isfinite(float(args.shared_residual_field_view_hinge_weight))
        or float(args.shared_residual_field_view_hinge_weight) < 0.0
    ):
        parser.error("--shared_residual_field_view_hinge_weight must be finite and >= 0")
    if int(args.shared_residual_field_view_hinge_min_samples) < 0:
        parser.error("--shared_residual_field_view_hinge_min_samples must be >= 0")
    if (
        not math.isfinite(float(args.shared_residual_field_duplicate_smooth_weight))
        or float(args.shared_residual_field_duplicate_smooth_weight) < 0.0
    ):
        parser.error("--shared_residual_field_duplicate_smooth_weight must be finite and >= 0")
    if (
        not math.isfinite(float(args.patch_cert_cluster_basis_max_fit_mse_regression))
        or float(args.patch_cert_cluster_basis_max_fit_mse_regression) < 0.0
    ):
        parser.error("--patch_cert_cluster_basis_max_fit_mse_regression must be finite and >= 0")
    if (
        not math.isfinite(float(args.patch_cert_cluster_basis_view_hinge_weight))
        or float(args.patch_cert_cluster_basis_view_hinge_weight) < 0.0
    ):
        parser.error("--patch_cert_cluster_basis_view_hinge_weight must be finite and >= 0")
    if int(args.patch_cert_cluster_basis_view_hinge_min_samples) < 0:
        parser.error("--patch_cert_cluster_basis_view_hinge_min_samples must be >= 0")
    if (
        not math.isfinite(float(args.patch_cert_cluster_basis_geometry_smooth_weight))
        or float(args.patch_cert_cluster_basis_geometry_smooth_weight) < 0.0
    ):
        parser.error("--patch_cert_cluster_basis_geometry_smooth_weight must be finite and >= 0")
    if int(args.patch_cert_carrier_holdout_groups) < 2:
        parser.error("--patch_cert_carrier_holdout_groups must be >= 2")
    if int(args.patch_cert_carrier_holdout_min_passing_groups) < 0:
        parser.error("--patch_cert_carrier_holdout_min_passing_groups must be >= 0")
    if int(args.patch_cert_carrier_holdout_min_passing_groups) > int(args.patch_cert_carrier_holdout_groups):
        parser.error("--patch_cert_carrier_holdout_min_passing_groups must be <= --patch_cert_carrier_holdout_groups")
    if int(args.patch_cert_carrier_holdout_min_group_samples) < 0:
        parser.error("--patch_cert_carrier_holdout_min_group_samples must be >= 0")
    if (
        not math.isfinite(float(args.patch_cert_carrier_holdout_max_mse_regression))
        or float(args.patch_cert_carrier_holdout_max_mse_regression) < 0.0
    ):
        parser.error("--patch_cert_carrier_holdout_max_mse_regression must be finite and >= 0")
    if (
        not math.isfinite(float(args.patch_cert_carrier_holdout_cvar_fraction))
        or float(args.patch_cert_carrier_holdout_cvar_fraction) <= 0.0
        or float(args.patch_cert_carrier_holdout_cvar_fraction) > 1.0
    ):
        parser.error("--patch_cert_carrier_holdout_cvar_fraction must be in (0, 1]")
    if not math.isfinite(float(args.patch_cert_carrier_holdout_cvar_weight)) or float(
        args.patch_cert_carrier_holdout_cvar_weight
    ) < 0.0:
        parser.error("--patch_cert_carrier_holdout_cvar_weight must be finite and >= 0")
    if int(args.patch_cert_carrier_holdout_max_carriers) < 0:
        parser.error("--patch_cert_carrier_holdout_max_carriers must be >= 0")
    if not math.isfinite(float(args.validation_gain_max_scale)) or float(args.validation_gain_max_scale) < 1.0:
        parser.error("--validation_gain_max_scale must be finite and >= 1")
    if int(args.patch_cert_carrier_holdout_auto_prefix_min_faces) < 0:
        parser.error("--patch_cert_carrier_holdout_auto_prefix_min_faces must be >= 0")
    if not math.isfinite(float(args.patch_cert_carrier_holdout_auto_prefix_face_bonus)) or float(
        args.patch_cert_carrier_holdout_auto_prefix_face_bonus
    ) < 0.0:
        parser.error("--patch_cert_carrier_holdout_auto_prefix_face_bonus must be finite and >= 0")
    if bool(args.patch_cert_carrier_holdout_disjoint) and str(args.patch_cert_carrier_holdout_grouping) != "sample_balanced":
        parser.error(
            "--patch_cert_carrier_holdout_disjoint currently requires "
            "--patch_cert_carrier_holdout_grouping sample_balanced"
        )
    return args


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def read_selected_faces(
    csv_path: Path,
    *,
    top_k: int,
    min_view_hits: int,
    min_consistency: float,
    min_pixel_count: float,
) -> tuple[list[int], dict[int, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view_hits = int(_float(row, "view_hits"))
            consistency = _float(row, "residual_consistency")
            pixel_count = _float(row, "pixel_count")
            if view_hits < int(min_view_hits):
                continue
            if consistency < float(min_consistency):
                continue
            if pixel_count < float(min_pixel_count):
                continue
            rows.append(
                {
                    "face_id": int(_float(row, "face_id")),
                    "score": _float(row, "score"),
                    "pixel_count": pixel_count,
                    "view_hits": view_hits,
                    "consistency": consistency,
                    "mean_l1_error": _float(row, "mean_l1_error"),
                    "mean_residual_r": _float(row, "mean_residual_r"),
                    "mean_residual_g": _float(row, "mean_residual_g"),
                    "mean_residual_b": _float(row, "mean_residual_b"),
                }
            )
    rows.sort(key=lambda r: (float(r["score"]), float(r["pixel_count"])), reverse=True)
    rows = rows[: int(top_k)]
    stats = {
        int(row["face_id"]): {
            "score": float(row["score"]),
            "pixel_count": float(row["pixel_count"]),
            "view_hits": float(row["view_hits"]),
            "consistency": float(row["consistency"]),
            "mean_l1_error": float(row["mean_l1_error"]),
            "mean_residual_r": float(row["mean_residual_r"]),
            "mean_residual_g": float(row["mean_residual_g"]),
            "mean_residual_b": float(row["mean_residual_b"]),
        }
        for row in rows
    }
    return [int(row["face_id"]) for row in rows], stats


def split_view_paths(view_paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    if len(view_paths) < 3:
        return view_paths, view_paths
    stride = max(int(stride), 2)
    fit: list[Path] = []
    val: list[Path] = []
    for idx, path in enumerate(view_paths):
        if idx % stride == 0:
            val.append(path)
        else:
            fit.append(path)
    if not fit or not val:
        return view_paths, view_paths
    return fit, val


def load_region_carrier_index(path: Path | None) -> dict[str, dict[int, list[tuple[int, int, int, int]]]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"region carrier JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    carriers = payload.get("carriers", [])
    if not isinstance(carriers, list):
        return {}
    index: dict[str, dict[int, list[tuple[int, int, int, int]]]] = {}
    for carrier in carriers:
        if not isinstance(carrier, dict):
            continue
        regions = carrier.get("regions", [])
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            view = str(region.get("view", "")).strip()
            if not view:
                continue
            view = Path(view).stem
            bbox = region.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                box = tuple(int(round(float(v))) for v in bbox)
            except Exception:
                continue
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            faces = region.get("face_ids", [])
            if not isinstance(faces, list):
                continue
            view_index = index.setdefault(view, {})
            for fid_raw in faces:
                try:
                    fid = int(fid_raw)
                except Exception:
                    continue
                view_index.setdefault(fid, []).append((x0, y0, x1, y1))
    return index


def region_bins_for_samples(
    region_index: dict[str, dict[int, list[tuple[int, int, int, int]]]],
    *,
    view_name: str,
    face_id: int,
    xs: np.ndarray,
    ys: np.ndarray,
    boundary_px: int,
) -> np.ndarray:
    bins = np.zeros((int(xs.shape[0]),), dtype=np.uint8)
    if not region_index:
        return bins
    boxes = region_index.get(str(view_name), {}).get(int(face_id), [])
    if not boxes:
        return bins
    bins.fill(1)
    margin = int(boundary_px)
    for x0, y0, x1, y1 in boxes:
        inside = (xs >= x0 - margin) & (xs < x1 + margin) & (ys >= y0 - margin) & (ys < y1 + margin)
        bins[inside] = 2
    return bins


def summarize_region_bins(samples: PixelSamples) -> dict[str, Any]:
    if samples.count == 0:
        return {"outside": 0, "context": 0, "core": 0, "total": 0, "core_fraction": 0.0}
    bins = samples.region_bins.astype(np.uint8, copy=False).reshape(-1)
    outside = int((bins == 0).sum())
    context = int((bins == 1).sum())
    core = int((bins == 2).sum())
    total = int(bins.shape[0])
    return {
        "outside": outside,
        "context": context,
        "core": core,
        "total": total,
        "core_fraction": float(core) / max(float(total), 1.0),
    }


def collect_samples(
    view_paths: list[Path],
    selected_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    *,
    high_error_quantile: float,
    min_alpha: float,
    barycentric_tolerance: float,
    max_samples_per_face_view: int,
    max_total_samples: int,
    uniform_barycentric: bool,
    face_score_weight_power: float = 0.0,
    face_score_weight_max: float = 4.0,
    region_index: dict[str, dict[int, list[tuple[int, int, int, int]]]] | None = None,
    region_core_weight: float = 1.0,
    region_context_weight: float = 1.0,
    region_outside_weight: float = 1.0,
    region_boundary_px: int = 0,
) -> PixelSamples:
    selected = set(int(fid) for fid in selected_faces)
    face_chunks: list[np.ndarray] = []
    bary_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    center_chunks: list[np.ndarray] = []
    region_bin_chunks: list[np.ndarray] = []
    sample_view_names: list[str] = []
    remaining = int(max_total_samples)
    tol = float(barycentric_tolerance)
    score_power = max(float(face_score_weight_power), 0.0)
    score_weight_max = max(float(face_score_weight_max), 1.0)
    positive_scores = [
        max(float(face_stats.get(int(fid), {}).get("score", 0.0)), 0.0)
        for fid in selected_faces
    ]
    positive_scores = [score for score in positive_scores if score > 0.0]
    score_ref = float(np.median(np.asarray(positive_scores, dtype=np.float64))) if positive_scores else 1.0
    score_ref = max(score_ref, 1e-8)

    for view_path in view_paths:
        if remaining <= 0:
            break
        with np.load(view_path) as z:
            required = {"face_id", "residual_l1", "alpha", "residual_rgb", "camera_center"}
            if not bool(uniform_barycentric):
                required.update({"barycentric", "barycentric_valid"})
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(
                    f"{view_path} missing required face-local SH1 evidence fields: {missing}. "
                    "Rebuild the cache with barycentric maps or use --uniform_barycentric."
                )
            face_id = z["face_id"].astype(np.int64)
            residual_l1 = z["residual_l1"].astype(np.float32)
            alpha = z["alpha"].astype(np.float32)
            if alpha.ndim == 3:
                alpha = np.squeeze(alpha, axis=0)
            residual_rgb = z["residual_rgb"].astype(np.float32)
            if bool(uniform_barycentric):
                barycentric = np.empty((3,) + face_id.shape, dtype=np.float32)
                bary_valid = np.ones_like(face_id, dtype=bool)
            else:
                barycentric = z["barycentric"].astype(np.float32)
                bary_valid = z["barycentric_valid"].astype(bool)
            camera_center = z["camera_center"].astype(np.float32).reshape(3)

        threshold = float(np.quantile(residual_l1.reshape(-1), float(high_error_quantile)))
        base_valid = bary_valid & (residual_l1 >= threshold) & (alpha >= float(min_alpha))
        if not np.any(base_valid):
            continue
        present = sorted(set(int(x) for x in np.unique(face_id[base_valid])) & selected)
        if not present:
            continue
        for fid in present:
            if remaining <= 0:
                break
            mask = base_valid & (face_id == int(fid))
            if not np.any(mask):
                continue
            if bool(uniform_barycentric):
                b = np.full((int(mask.sum()), 3), 1.0 / 3.0, dtype=np.float32)
            else:
                b = barycentric[:, mask].T.astype(np.float32)
            inside = np.all((b >= -tol) & (b <= 1.0 + tol), axis=1)
            if not np.any(inside):
                continue
            ys, xs = np.nonzero(mask)
            ys = ys[inside]
            xs = xs[inside]
            b = b[inside]
            b = np.clip(b, 0.0, 1.0)
            b = b / np.maximum(b.sum(axis=1, keepdims=True), 1e-8)
            n = int(b.shape[0])
            if n <= 0:
                continue
            cap = min(int(max_samples_per_face_view), remaining, n)
            if n > cap:
                take = np.linspace(0, n - 1, cap, dtype=np.int64)
                ys = ys[take]
                xs = xs[take]
                b = b[take]
                n = cap
            residual = residual_rgb[:, ys, xs].T.astype(np.float32)
            l1 = residual_l1[ys, xs].astype(np.float32)
            stat = face_stats.get(int(fid), {})
            consistency = float(stat.get("consistency", 1.0))
            score_weight = 1.0
            if score_power > 0.0:
                face_score = max(float(stat.get("score", score_ref)), 0.0)
                score_weight = float(np.clip((face_score / score_ref) ** score_power, 1e-3, score_weight_max))
            region_bins = region_bins_for_samples(
                region_index or {},
                view_name=view_path.stem,
                face_id=int(fid),
                xs=xs,
                ys=ys,
                boundary_px=int(region_boundary_px),
            )
            region_weights = np.full((n,), float(region_outside_weight), dtype=np.float32)
            region_weights[region_bins == 1] = float(region_context_weight)
            region_weights[region_bins == 2] = float(region_core_weight)
            weights = np.maximum(l1, 1e-4) * max(consistency, 1e-3) * score_weight * region_weights
            face_chunks.append(np.full((n,), int(fid), dtype=np.int64))
            bary_chunks.append(b.astype(np.float32))
            residual_chunks.append(residual.astype(np.float32))
            weight_chunks.append(weights.astype(np.float32))
            center_chunks.append(np.repeat(camera_center[None, :], n, axis=0).astype(np.float32))
            region_bin_chunks.append(region_bins.astype(np.uint8))
            sample_view_names.extend([view_path.stem] * n)
            remaining -= n

    if not face_chunks:
        empty = np.empty((0,), dtype=np.int64)
        return PixelSamples(
            face_ids=empty,
            barycentric=np.empty((0, 3), dtype=np.float32),
            residual_rgb=np.empty((0, 3), dtype=np.float32),
            weights=np.empty((0,), dtype=np.float32),
            camera_centers=np.empty((0, 3), dtype=np.float32),
            view_names=[],
            region_bins=np.empty((0,), dtype=np.uint8),
        )
    return PixelSamples(
        face_ids=np.concatenate(face_chunks),
        barycentric=np.concatenate(bary_chunks),
        residual_rgb=np.concatenate(residual_chunks),
        weights=np.concatenate(weight_chunks),
        camera_centers=np.concatenate(center_chunks),
        view_names=sample_view_names,
        region_bins=np.concatenate(region_bin_chunks),
    )


def subset_pixel_samples(samples: PixelSamples, mask: np.ndarray) -> PixelSamples:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if samples.count != int(mask.shape[0]):
        raise ValueError("sample subset mask length mismatch")
    view_np = np.asarray(samples.view_names, dtype=object)
    return PixelSamples(
        face_ids=samples.face_ids[mask],
        barycentric=samples.barycentric[mask],
        residual_rgb=samples.residual_rgb[mask],
        weights=samples.weights[mask],
        camera_centers=samples.camera_centers[mask],
        view_names=[str(v) for v in view_np[mask].tolist()],
        region_bins=samples.region_bins[mask],
    )


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def surface_edges(faces_local: torch.Tensor) -> torch.Tensor:
    if faces_local.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    edges = torch.cat([faces_local[:, [0, 1]], faces_local[:, [1, 2]], faces_local[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    return torch.unique(edges, dim=0)


def localize_samples(
    faces: torch.Tensor,
    selected_faces: list[int],
    samples: PixelSamples,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_vertex_ids = faces[torch.as_tensor(selected_faces, dtype=torch.long)].long().reshape(-1)
    selected_faces_local = torch.arange(len(selected_faces) * 3, dtype=torch.long).reshape(-1, 3)
    face_to_local = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    sample_faces_local = torch.as_tensor([face_to_local[int(fid)] for fid in samples.face_ids], dtype=torch.long)
    sample_vertex_ids = selected_faces_local[sample_faces_local]
    return source_vertex_ids, selected_faces_local, sample_vertex_ids


def _sh_basis(
    vertices_local: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    bary: torch.Tensor,
    camera_centers: torch.Tensor,
    *,
    degree: int,
) -> torch.Tensor:
    degree = int(degree)
    basis_count = (degree + 1) ** 2
    if sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3, basis_count), dtype=torch.float32, device=vertices_local.device)
    vpos = vertices_local[sample_vertex_ids]
    dirs = vpos - camera_centers[:, None, :]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    terms = [
        torch.full_like(x, float(C0)),
        -float(C1) * y,
        float(C1) * z,
        -float(C1) * x,
    ]
    if degree >= 2:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        terms.extend(
            [
                float(C2[0]) * xy,
                float(C2[1]) * yz,
                float(C2[2]) * (2.0 * zz - xx - yy),
                float(C2[3]) * xz,
                float(C2[4]) * (xx - yy),
            ]
        )
    if degree >= 3:
        terms.extend(
            [
                float(C3[0]) * y * (3.0 * xx - yy),
                float(C3[1]) * xy * z,
                float(C3[2]) * y * (4.0 * zz - xx - yy),
                float(C3[3]) * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                float(C3[4]) * x * (4.0 * zz - xx - yy),
                float(C3[5]) * z * (xx - yy),
                float(C3[6]) * x * (xx - 3.0 * yy),
            ]
        )
    basis = torch.stack(terms, dim=-1)
    return basis * bary[:, :, None]


def _predict(coeff: torch.Tensor, sample_vertex_ids: torch.Tensor, weighted_basis: torch.Tensor) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=coeff.device)
    sample_coeff = coeff[sample_vertex_ids]
    return (sample_coeff * weighted_basis[:, :, :, None]).sum(dim=(1, 2))


def _weighted_mse(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if sample_vertex_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=coeff.device)
    pred = _predict(coeff, sample_vertex_ids, weighted_basis)
    return (((pred - target) ** 2) * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)


def evaluate_proxy(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    if sample_vertex_ids.numel() == 0:
        return {
            "samples": 0,
            "mse_before": 0.0,
            "mse_after": 0.0,
            "relative_gain": 0.0,
            "mae_before": 0.0,
            "mae_after": 0.0,
        }
    zero = torch.zeros_like(coeff)
    with torch.no_grad():
        mse_before = _weighted_mse(zero, sample_vertex_ids, weighted_basis, target, weights)
        mse_after = _weighted_mse(coeff, sample_vertex_ids, weighted_basis, target, weights)
        pred_after = _predict(coeff, sample_vertex_ids, weighted_basis)
        mae_before = (target.abs() * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)
        mae_after = ((pred_after - target).abs() * weights[:, None]).sum() / (weights.sum().clamp_min(1e-8) * 3.0)
        gain = (mse_before - mse_after) / mse_before.clamp_min(1e-12)
    return {
        "samples": int(sample_vertex_ids.shape[0]),
        "mse_before": float(mse_before.detach().cpu().item()),
        "mse_after": float(mse_after.detach().cpu().item()),
        "relative_gain": float(gain.detach().cpu().item()),
        "mae_before": float(mae_before.detach().cpu().item()),
        "mae_after": float(mae_after.detach().cpu().item()),
    }


def evaluate_proxy_by_face(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    if sample_vertex_ids.numel() == 0:
        return {}
    with torch.no_grad():
        pred = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    weight_np = weights.detach().cpu().numpy().reshape(-1)
    face_np = sample_face_ids.astype(np.int64, copy=False).reshape(-1)
    out: dict[int, dict[str, float]] = {}
    for fid in np.unique(face_np).tolist():
        mask = face_np == int(fid)
        if not np.any(mask):
            continue
        w = weight_np[mask].reshape(-1, 1)
        y = target_np[mask]
        p = pred[mask]
        denom = float(max(float(w.sum()) * 3.0, 1e-8))
        mse_before = float(((y**2) * w).sum() / denom)
        mse_after = float((((p - y) ** 2) * w).sum() / denom)
        mae_before = float((np.abs(y) * w).sum() / denom)
        mae_after = float((np.abs(p - y) * w).sum() / denom)
        out[int(fid)] = {
            "samples": int(mask.sum()),
            "mse_before": mse_before,
            "mse_after": mse_after,
            "relative_gain": float((mse_before - mse_after) / max(mse_before, 1e-12)),
            "mae_before": mae_before,
            "mae_after": mae_after,
        }
    return out


def face_view_consensus_report(
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_consensus: float,
    min_views: int,
    min_view_samples: int,
    min_cosine: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = float(min_consensus) > 0.0
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "min_face_view_consensus": float(min_consensus),
        "min_face_consensus_views": int(min_views),
        "min_face_consensus_view_samples": int(min_view_samples),
        "face_consensus_min_cosine": float(min_cosine),
        "faces_evaluated": 0,
        "faces_passing": 0,
    }
    if not enabled:
        return {}, summary
    if samples.count == 0:
        return {}, summary

    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    view_np = np.asarray(samples.view_names, dtype=object)
    if target_np.shape[0] != face_np.shape[0] or view_np.shape[0] != face_np.shape[0]:
        raise ValueError("sample face/view arrays do not match target shape")

    required_views = max(int(min_views), 1)
    required_view_samples = max(int(min_view_samples), 1)
    out: dict[int, dict[str, Any]] = {}
    for fid in np.unique(face_np).tolist():
        face_mask = face_np == int(fid)
        view_vectors: list[np.ndarray] = []
        view_names: list[str] = []
        view_sample_counts: list[int] = []
        for view_name in sorted(set(str(v) for v in view_np[face_mask].tolist())):
            view_mask = face_mask & (view_np == view_name)
            sample_count = int(view_mask.sum())
            if sample_count < required_view_samples:
                continue
            w = weight_np[view_mask].reshape(-1, 1)
            denom = max(float(w.sum()), 1e-8)
            vector = (target_np[view_mask] * w).sum(axis=0) / denom
            if float(np.linalg.norm(vector)) <= 1e-10:
                continue
            view_vectors.append(vector.astype(np.float32, copy=False))
            view_names.append(view_name)
            view_sample_counts.append(sample_count)
        if view_vectors:
            vectors = np.stack(view_vectors, axis=0)
            direction = vectors.mean(axis=0)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-10:
                norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-10)
                cosines = (vectors @ direction) / (norms * direction_norm)
                agreeing = int(np.sum(cosines >= float(min_cosine)))
            else:
                cosines = np.zeros((len(view_vectors),), dtype=np.float32)
                agreeing = 0
        else:
            direction_norm = 0.0
            cosines = np.empty((0,), dtype=np.float32)
            agreeing = 0
        view_count = int(len(view_vectors))
        consensus = float(agreeing / max(view_count, 1))
        passed = bool(view_count >= required_views and consensus >= float(min_consensus))
        out[int(fid)] = {
            "view_count": view_count,
            "agreeing_views": agreeing,
            "consensus": consensus,
            "residual_norm": float(direction_norm),
            "mean_cosine": float(np.mean(cosines)) if cosines.size else 0.0,
            "min_cosine": float(np.min(cosines)) if cosines.size else 0.0,
            "passed": passed,
            "view_names": view_names[:16],
            "view_sample_counts": view_sample_counts[:16],
        }

    summary["faces_evaluated"] = int(len(out))
    summary["faces_passing"] = int(sum(1 for row in out.values() if bool(row.get("passed", False))))
    return out, summary


def face_view_gain_certificate_report(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    min_views: int,
    min_relative_gain: float,
    min_view_samples: int,
    min_fraction: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enabled = int(min_views) > 0
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "min_face_gain_certificate_views": int(min_views),
        "min_face_gain_certificate_relative_gain": float(min_relative_gain),
        "min_face_gain_certificate_view_samples": int(min_view_samples),
        "min_face_gain_certificate_fraction": float(min_fraction),
        "faces_evaluated": 0,
        "faces_passing": 0,
        "eligible_views": 0,
        "beneficial_views": 0,
        "mean_beneficial_fraction": 0.0,
    }
    if not enabled:
        return {}, summary
    if samples.count == 0 or sample_vertex_ids.numel() == 0:
        return {}, summary

    with torch.no_grad():
        pred_np = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    view_np = np.asarray(samples.view_names, dtype=object)
    if (
        pred_np.shape[0] != target_np.shape[0]
        or target_np.shape[0] != face_np.shape[0]
        or view_np.shape[0] != face_np.shape[0]
    ):
        raise ValueError("sample prediction/target/face/view arrays do not match")

    required_views = max(int(min_views), 1)
    required_view_samples = max(int(min_view_samples), 1)
    required_fraction = max(float(min_fraction), 0.0)
    out: dict[int, dict[str, Any]] = {}
    beneficial_fractions: list[float] = []
    for fid in np.unique(face_np).tolist():
        face_mask = face_np == int(fid)
        view_rows: list[dict[str, Any]] = []
        for view_name in sorted(set(str(v) for v in view_np[face_mask].tolist())):
            view_mask = face_mask & (view_np == view_name)
            sample_count = int(view_mask.sum())
            if sample_count < required_view_samples:
                continue
            w = weight_np[view_mask].reshape(-1, 1)
            y = target_np[view_mask]
            p = pred_np[view_mask]
            denom = max(float(w.sum()) * 3.0, 1e-8)
            mse_before = float(((y**2) * w).sum() / denom)
            mse_after = float((((p - y) ** 2) * w).sum() / denom)
            relative_gain = float((mse_before - mse_after) / max(mse_before, 1e-12))
            view_rows.append(
                {
                    "view_name": view_name,
                    "samples": sample_count,
                    "mse_before": mse_before,
                    "mse_after": mse_after,
                    "relative_gain": relative_gain,
                    "passed": bool(relative_gain >= float(min_relative_gain)),
                }
            )

        eligible = int(len(view_rows))
        beneficial = int(sum(1 for row in view_rows if bool(row["passed"])))
        fraction = float(beneficial / max(eligible, 1))
        passed = bool(eligible >= required_views and beneficial >= required_views and fraction >= required_fraction)
        gains = [float(row["relative_gain"]) for row in view_rows]
        beneficial_fractions.append(fraction)
        out[int(fid)] = {
            "eligible_view_count": eligible,
            "beneficial_view_count": beneficial,
            "beneficial_fraction": fraction,
            "min_relative_gain": float(min(gains)) if gains else 0.0,
            "mean_relative_gain": float(np.mean(gains)) if gains else 0.0,
            "max_relative_gain": float(max(gains)) if gains else 0.0,
            "passed": passed,
            "views": view_rows[:16],
        }

    summary["faces_evaluated"] = int(len(out))
    summary["faces_passing"] = int(sum(1 for row in out.values() if bool(row.get("passed", False))))
    summary["eligible_views"] = int(sum(int(row.get("eligible_view_count", 0)) for row in out.values()))
    summary["beneficial_views"] = int(sum(int(row.get("beneficial_view_count", 0)) for row in out.values()))
    summary["mean_beneficial_fraction"] = float(np.mean(beneficial_fractions)) if beneficial_fractions else 0.0
    return out, summary


def calibrate_coeff_by_policy_val(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    samples: PixelSamples,
    target: torch.Tensor,
    weights: torch.Tensor,
    selected_faces: list[int],
    *,
    mode: str,
    min_samples: int,
    max_gain_scale: float,
) -> tuple[torch.Tensor, dict[str, Any], dict[int, dict[str, Any]]]:
    mode = str(mode)
    gain_mode = mode in {"global_gain", "face_gain"}
    upper_scale = float(max(max_gain_scale if gain_mode else 1.0, 1.0))
    summary: dict[str, Any] = {
        "mode": mode,
        "enabled": mode != "none",
        "min_samples": int(min_samples),
        "gain_enabled": bool(gain_mode),
        "max_gain_scale": float(upper_scale),
        "global_scale": 1.0,
        "faces_evaluated": 0,
        "faces_scaled": 0,
        "zero_scale_faces": 0,
        "mean_scale": 1.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
    }
    if mode == "none":
        return coeff, summary, {}
    if samples.count == 0 or sample_vertex_ids.numel() == 0 or coeff.numel() == 0:
        if mode in {"global", "global_gain"}:
            return coeff * 0.0, {**summary, "global_scale": 0.0, "mean_scale": 0.0, "min_scale": 0.0, "max_scale": 0.0}, {}
        return coeff * 0.0, {**summary, "mean_scale": 0.0, "min_scale": 0.0, "max_scale": 0.0}, {}

    with torch.no_grad():
        pred_np = _predict(coeff, sample_vertex_ids, weighted_basis).detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    weight_np = weights.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 1)
    face_np = samples.face_ids.astype(np.int64, copy=False).reshape(-1)
    if pred_np.shape[0] != target_np.shape[0] or face_np.shape[0] != target_np.shape[0]:
        raise ValueError("sample prediction/target/face arrays do not match for validation shrink")

    def fit_scale(mask: np.ndarray) -> tuple[float, int, float]:
        sample_count = int(mask.sum())
        if sample_count < max(int(min_samples), 1):
            return 0.0, sample_count, 0.0
        p = pred_np[mask]
        y = target_np[mask]
        w = weight_np[mask]
        numerator = float((w * p * y).sum())
        denominator = float((w * p * p).sum())
        if denominator <= 1e-12:
            return 0.0, sample_count, 0.0
        raw_scale = numerator / denominator
        scale = float(min(max(raw_scale, 0.0), upper_scale))
        return scale, sample_count, float(raw_scale)

    if mode in {"global", "global_gain"}:
        scale, sample_count, raw_scale = fit_scale(np.ones((target_np.shape[0],), dtype=bool))
        out = coeff * float(scale)
        summary.update(
            {
                "global_scale": scale,
                "samples": sample_count,
                "raw_global_scale": raw_scale,
                "mean_scale": scale,
                "min_scale": scale,
                "max_scale": scale,
                "faces_evaluated": int(len(np.unique(face_np))) if face_np.size else 0,
                "faces_scaled": int(len(np.unique(face_np))) if scale < 0.999999 and face_np.size else 0,
                "zero_scale_faces": int(len(np.unique(face_np))) if scale <= 1e-8 and face_np.size else 0,
            }
        )
        return out, summary, {}

    if mode not in {"face", "face_gain"}:
        raise ValueError(f"unsupported validation shrink mode: {mode}")

    out = coeff.clone()
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    per_face: dict[int, dict[str, Any]] = {}
    scales: list[float] = []
    for fid in selected_faces:
        face_id = int(fid)
        scale, sample_count, raw_scale = fit_scale(face_np == face_id)
        scales.append(scale)
        row = face_to_selected[face_id]
        out[row * 3 : row * 3 + 3] = out[row * 3 : row * 3 + 3] * float(scale)
        per_face[face_id] = {
            "scale": float(scale),
            "raw_scale": float(raw_scale),
            "samples": int(sample_count),
            "passed_min_samples": bool(sample_count >= max(int(min_samples), 1)),
        }

    if scales:
        scale_np = np.asarray(scales, dtype=np.float32)
        summary.update(
            {
                "faces_evaluated": int(len(scales)),
                "faces_scaled": int(np.sum(scale_np < 0.999999)),
                "zero_scale_faces": int(np.sum(scale_np <= 1e-8)),
                "mean_scale": float(scale_np.mean()),
                "min_scale": float(scale_np.min()),
                "max_scale": float(scale_np.max()),
            }
        )
    return out, summary, per_face


def summarize_crossfold_face_gain(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    vertices: torch.Tensor,
    view_paths: list[Path],
    face_stats: dict[int, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    holdout_samples: PixelSamples | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    folds = int(args.crossfold_gain_certificate_folds)
    has_reserved_holdout = holdout_samples is not None
    summary: dict[str, Any] = {
        "enabled": bool(folds > 1),
        "certificate_type": (
            "reserved_policy_val_sample_fold_consistency"
            if has_reserved_holdout
            else "all_train_fold_consistency_not_crossfit"
        ),
        "folds": max(folds, 0),
        "fold_grouping": "sample_balanced" if has_reserved_holdout else "view_modulo",
        "reserved_holdout_samples": int(holdout_samples.count) if holdout_samples is not None else 0,
        "min_passing_folds": int(args.crossfold_min_passing_folds),
        "min_fold_relative_gain": float(args.crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.crossfold_min_fold_samples),
        "faces_evaluated": 0,
        "faces_passing": 0,
        "fold_summaries": [],
    }
    if folds <= 1:
        return {}, summary
    required_passing = int(args.crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = folds
    summary["min_passing_folds"] = int(required_passing)
    if not selected_faces or coeff.numel() == 0 or (not view_paths and holdout_samples is None):
        return {}, summary

    per_face_rows: dict[int, dict[str, Any]] = {
        int(fid): {
            "passing_folds": 0,
            "eligible_folds": 0,
            "folds": [],
        }
        for fid in selected_faces
    }
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    reserved_indices = (
        np.arange(int(holdout_samples.count), dtype=np.int64)
        if holdout_samples is not None
        else np.empty((0,), dtype=np.int64)
    )
    for fold_idx in range(folds):
        if holdout_samples is not None:
            fold_mask = (reserved_indices % folds) == int(fold_idx)
            fold_samples = subset_pixel_samples(holdout_samples, fold_mask)
            fold_view_names = sorted(set(str(v) for v in fold_samples.view_names))
        else:
            fold_paths = [path for idx, path in enumerate(view_paths) if idx % folds == fold_idx]
            fold_samples = collect_samples(
                fold_paths,
                selected_faces,
                face_stats,
                high_error_quantile=float(args.high_error_quantile),
                min_alpha=float(args.min_alpha),
                barycentric_tolerance=float(args.barycentric_tolerance),
                max_samples_per_face_view=int(args.max_samples_per_face_view),
                max_total_samples=max(int(args.max_total_samples // max(folds, 1)), 1),
                uniform_barycentric=bool(args.uniform_barycentric),
                face_score_weight_power=float(args.face_score_weight_power),
                face_score_weight_max=float(args.face_score_weight_max),
                region_index=load_region_carrier_index(args.region_carrier_json),
                region_core_weight=float(args.region_core_weight),
                region_context_weight=float(args.region_context_weight),
                region_outside_weight=float(args.region_outside_weight),
                region_boundary_px=int(args.region_boundary_px),
            )
            fold_view_names = [p.stem for p in fold_paths]
        if fold_samples.count:
            _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
        else:
            fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
            fold_samples,
            fold_sample_vertex_ids,
            vertices_local,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            sh_degree=int(args.sh_degree),
            device=device,
        )
        fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
        fold_face = evaluate_proxy_by_face(coeff, fold_ids, fold_basis, fold_target, fold_weights, fold_samples.face_ids)
        fold_passing_faces = 0
        for fid in selected_faces:
            stats = fold_face.get(int(fid), {})
            samples = int(stats.get("samples", 0))
            relative_gain = float(stats.get("relative_gain", -1.0))
            eligible = samples >= int(args.crossfold_min_fold_samples)
            passed = bool(eligible and relative_gain >= float(args.crossfold_min_fold_relative_gain))
            row = per_face_rows[int(fid)]
            if eligible:
                row["eligible_folds"] += 1
            if passed:
                row["passing_folds"] += 1
                fold_passing_faces += 1
            row["folds"].append(
                {
                    "fold": int(fold_idx),
                    "samples": samples,
                    "relative_gain": relative_gain,
                    "eligible": bool(eligible),
                    "passed": bool(passed),
                }
            )
        summary["fold_summaries"].append(
            {
                "fold": int(fold_idx),
                "view_names": fold_view_names[:16],
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
                "passing_faces": int(fold_passing_faces),
            }
        )

    for fid, row in per_face_rows.items():
        gains = [float(fold["relative_gain"]) for fold in row["folds"] if bool(fold["eligible"])]
        row["passed"] = bool(int(row["passing_folds"]) >= required_passing)
        row["min_relative_gain"] = float(min(gains)) if gains else 0.0
        row["mean_relative_gain"] = float(np.mean(gains)) if gains else 0.0
    summary["faces_evaluated"] = int(len(per_face_rows))
    summary["faces_passing"] = int(sum(1 for row in per_face_rows.values() if bool(row.get("passed", False))))
    return per_face_rows, summary


def face_residual_direction(face_stats: dict[int, dict[str, float]], face_id: int) -> np.ndarray:
    stats = face_stats.get(int(face_id), {})
    vec = np.asarray(
        [
            float(stats.get("mean_residual_r", 0.0)),
            float(stats.get("mean_residual_g", 0.0)),
            float(stats.get("mean_residual_b", 0.0)),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return vec / norm


def residual_direction_cosine(face_stats: dict[int, dict[str, float]], a: int, b: int) -> float:
    da = face_residual_direction(face_stats, int(a))
    db = face_residual_direction(face_stats, int(b))
    if float(np.linalg.norm(da)) <= 1e-8 or float(np.linalg.norm(db)) <= 1e-8:
        return 0.0
    return float(np.clip(float(np.dot(da, db)), -1.0, 1.0))


def selected_face_adjacency(faces: torch.Tensor, selected_faces: list[int]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {int(fid): set() for fid in selected_faces}
    vertex_to_faces: dict[int, list[int]] = {}
    for fid in selected_faces:
        face_id = int(fid)
        if face_id < 0 or face_id >= int(faces.shape[0]):
            continue
        for vertex_id in faces[face_id].detach().cpu().long().tolist():
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)
    for incident in vertex_to_faces.values():
        if len(incident) < 2:
            continue
        for fid in incident:
            row = adjacency.setdefault(int(fid), set())
            for other in incident:
                if int(other) != int(fid):
                    row.add(int(other))
    return adjacency


def selected_face_centers(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    selected_faces: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if not selected_faces:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), {}
    face_ids = np.asarray([int(fid) for fid in selected_faces], dtype=np.int64)
    valid = (face_ids >= 0) & (face_ids < int(faces.shape[0]))
    face_ids = face_ids[valid]
    if face_ids.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float32), {}
    face_tensor = faces[torch.as_tensor(face_ids, dtype=torch.long)].detach().cpu().long()
    centers = vertices[face_tensor].detach().cpu().float().mean(dim=1).numpy().astype(np.float32, copy=False)
    return face_ids, centers, {int(fid): idx for idx, fid in enumerate(face_ids.tolist())}


def centroid_neighbor_candidates(
    seed: int,
    face_ids: np.ndarray,
    centers: np.ndarray,
    center_index: dict[int, int],
    max_candidates: int,
) -> list[int]:
    idx = center_index.get(int(seed))
    if idx is None or centers.shape[0] <= 1:
        return []
    delta = centers - centers[idx : idx + 1]
    dist2 = np.sum(delta * delta, axis=1)
    out: list[int] = []
    for j in np.argsort(dist2):
        fid = int(face_ids[int(j)])
        if fid == int(seed):
            continue
        out.append(fid)
        if len(out) >= int(max_candidates):
            break
    return out


def evaluate_proxy_for_faces(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
    face_ids: list[int],
) -> dict[str, float]:
    if not face_ids or sample_vertex_ids.numel() == 0:
        return evaluate_proxy(
            coeff,
            sample_vertex_ids[:0],
            weighted_basis[:0],
            target[:0],
            weights[:0],
        )
    mask = np.isin(sample_face_ids.astype(np.int64, copy=False), np.asarray(face_ids, dtype=np.int64))
    if not np.any(mask):
        return evaluate_proxy(
            coeff,
            sample_vertex_ids[:0],
            weighted_basis[:0],
            target[:0],
            weights[:0],
        )
    idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long, device=sample_vertex_ids.device)
    return evaluate_proxy(coeff, sample_vertex_ids[idx], weighted_basis[idx], target[idx], weights[idx])


def build_patch_crossfold_cache(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    vertices: torch.Tensor,
    view_paths: list[Path],
    face_stats: dict[int, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    holdout_samples: PixelSamples | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds = int(args.patch_cert_crossfold_folds)
    summary: dict[str, Any] = {
        "enabled": bool(folds > 1),
        "certificate_type": "all_train_patch_fold_consistency_not_crossfit",
        "folds": max(folds, 0),
        "min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "fold_summaries": [],
    }
    if folds <= 1 or not selected_faces or not view_paths or coeff.numel() == 0:
        return [], summary
    required_passing = int(args.patch_cert_crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = folds
    summary["min_passing_folds"] = int(required_passing)
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    cache: list[dict[str, Any]] = []
    for fold_idx in range(folds):
        fold_paths = [path for idx, path in enumerate(view_paths) if idx % folds == fold_idx]
        fold_samples = collect_samples(
            fold_paths,
            selected_faces,
            face_stats,
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            barycentric_tolerance=float(args.barycentric_tolerance),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            max_total_samples=max(int(args.max_total_samples // max(folds, 1)), 1),
            uniform_barycentric=bool(args.uniform_barycentric),
            face_score_weight_power=float(args.face_score_weight_power),
            face_score_weight_max=float(args.face_score_weight_max),
            region_index=load_region_carrier_index(args.region_carrier_json),
            region_core_weight=float(args.region_core_weight),
            region_context_weight=float(args.region_context_weight),
            region_outside_weight=float(args.region_outside_weight),
            region_boundary_px=int(args.region_boundary_px),
        )
        if fold_samples.count:
            _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
        else:
            fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
            fold_samples,
            fold_sample_vertex_ids,
            vertices_local,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            sh_degree=int(args.sh_degree),
            device=device,
        )
        fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
        row = {
            "fold": int(fold_idx),
            "view_names": [p.stem for p in fold_paths],
            "samples": int(fold_samples.count),
            "proxy": fold_proxy,
            "ids": fold_ids,
            "basis": fold_basis,
            "target": fold_target,
            "weights": fold_weights,
            "face_ids": fold_samples.face_ids,
        }
        summary["fold_summaries"].append(
            {
                "fold": int(fold_idx),
                "view_names": row["view_names"],
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
            }
        )
        cache.append(row)
    return cache, summary


def patch_crossfold_certificate_for_faces(
    coeff: torch.Tensor,
    fold_cache: list[dict[str, Any]],
    face_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    folds = int(args.patch_cert_crossfold_folds)
    enabled = bool(folds > 1)
    required_passing = int(args.patch_cert_crossfold_min_passing_folds)
    if required_passing <= 0:
        required_passing = max(folds, 0)
    result: dict[str, Any] = {
        "enabled": enabled,
        "folds": max(folds, 0),
        "min_passing_folds": int(required_passing),
        "min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "passing_folds": 0,
        "eligible_folds": 0,
        "passed": not enabled,
        "fold_rows": [],
    }
    if not enabled:
        return result
    gains: list[float] = []
    for fold in fold_cache:
        proxy = evaluate_proxy_for_faces(
            coeff,
            fold["ids"],
            fold["basis"],
            fold["target"],
            fold["weights"],
            fold["face_ids"],
            face_ids,
        )
        samples = int(proxy.get("samples", 0))
        relative_gain = float(proxy.get("relative_gain", -1.0))
        eligible = samples >= int(args.patch_cert_crossfold_min_fold_samples)
        passed = bool(eligible and relative_gain >= float(args.patch_cert_crossfold_min_fold_relative_gain))
        if eligible:
            result["eligible_folds"] = int(result["eligible_folds"]) + 1
            gains.append(relative_gain)
        if passed:
            result["passing_folds"] = int(result["passing_folds"]) + 1
        result["fold_rows"].append(
            {
                "fold": int(fold.get("fold", len(result["fold_rows"]))),
                "samples": samples,
                "relative_gain": relative_gain,
                "eligible": bool(eligible),
                "passed": bool(passed),
            }
        )
    result["passed"] = bool(int(result["passing_folds"]) >= required_passing)
    result["min_relative_gain"] = float(min(gains)) if gains else 0.0
    result["mean_relative_gain"] = float(np.mean(np.asarray(gains, dtype=np.float32))) if gains else 0.0
    return result


def build_carrier_holdout_cache(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    vertices: torch.Tensor,
    view_paths: list[Path],
    face_stats: dict[int, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    holdout_samples: PixelSamples | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = int(args.patch_cert_carrier_holdout_groups)
    grouping = str(args.patch_cert_carrier_holdout_grouping)
    explicit_sample_holdout = holdout_samples is not None
    summary: dict[str, Any] = {
        "enabled": bool(args.patch_cert_carrier_holdout_selector),
        "certificate_type": (
            "train_only_whole_carrier_sample_holdout"
            if grouping == "sample_balanced"
            else "train_only_whole_carrier_view_holdout"
        ),
        "selection_unit": "patchcert_carrier",
        "test_usage": "none",
        "holdout_source": "policy_val_train_split",
        "grouping": grouping,
        "disjoint_from_policy_tuning": bool(args.patch_cert_carrier_holdout_disjoint),
        "explicit_holdout_samples": int(holdout_samples.count) if holdout_samples is not None else 0,
        "groups": int(groups),
        "source_view_count": int(len(view_paths)),
        "fold_summaries": [],
    }
    if not bool(args.patch_cert_carrier_holdout_selector):
        return [], summary
    if groups <= 1 or not selected_faces or coeff.numel() == 0 or (not view_paths and not explicit_sample_holdout):
        summary["blocked_reason"] = "missing_holdout_inputs"
        return [], summary

    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    cache: list[dict[str, Any]] = []
    if grouping == "sample_balanced":
        all_samples = holdout_samples
        if all_samples is None:
            all_samples = collect_samples(
                view_paths,
                selected_faces,
                face_stats,
                high_error_quantile=float(args.high_error_quantile),
                min_alpha=float(args.min_alpha),
                barycentric_tolerance=float(args.barycentric_tolerance),
                max_samples_per_face_view=int(args.max_samples_per_face_view),
                max_total_samples=int(args.max_total_samples),
                uniform_barycentric=bool(args.uniform_barycentric),
                face_score_weight_power=float(args.face_score_weight_power),
                face_score_weight_max=float(args.face_score_weight_max),
                region_index=load_region_carrier_index(args.region_carrier_json),
                region_core_weight=float(args.region_core_weight),
                region_context_weight=float(args.region_context_weight),
                region_outside_weight=float(args.region_outside_weight),
                region_boundary_px=int(args.region_boundary_px),
            )
        summary["sample_count"] = int(all_samples.count)
        sample_indices = np.arange(int(all_samples.count), dtype=np.int64)
        for group_idx in range(groups):
            group_mask = (sample_indices % groups) == int(group_idx)
            fold_samples = subset_pixel_samples(all_samples, group_mask)
            if fold_samples.count:
                _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
            else:
                fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
            fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
                fold_samples,
                fold_sample_vertex_ids,
                vertices_local,
                strength=float(args.strength),
                max_abs_delta_rgb=float(args.max_abs_delta_rgb),
                sh_degree=int(args.sh_degree),
                device=device,
            )
            fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
            view_names = sorted(set(str(v) for v in fold_samples.view_names))
            row = {
                "group": int(group_idx),
                "view_names": view_names,
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
                "ids": fold_ids,
                "basis": fold_basis,
                "target": fold_target,
                "weights": fold_weights,
                "face_ids": fold_samples.face_ids,
            }
            cache.append(row)
            summary["fold_summaries"].append(
                {
                    "group": int(group_idx),
                    "view_count": int(len(view_names)),
                    "view_names": view_names[:16],
                    "samples": int(fold_samples.count),
                    "proxy": fold_proxy,
                }
            )
        return cache, summary

    for group_idx in range(groups):
        fold_paths = [path for idx, path in enumerate(view_paths) if idx % groups == group_idx]
        fold_samples = collect_samples(
            fold_paths,
            selected_faces,
            face_stats,
            high_error_quantile=float(args.high_error_quantile),
            min_alpha=float(args.min_alpha),
            barycentric_tolerance=float(args.barycentric_tolerance),
            max_samples_per_face_view=int(args.max_samples_per_face_view),
            max_total_samples=max(int(args.max_total_samples // max(groups, 1)), 1),
            uniform_barycentric=bool(args.uniform_barycentric),
            face_score_weight_power=float(args.face_score_weight_power),
            face_score_weight_max=float(args.face_score_weight_max),
            region_index=load_region_carrier_index(args.region_carrier_json),
            region_core_weight=float(args.region_core_weight),
            region_context_weight=float(args.region_context_weight),
            region_outside_weight=float(args.region_outside_weight),
            region_boundary_px=int(args.region_boundary_px),
        )
        if fold_samples.count:
            _, _, fold_sample_vertex_ids = localize_samples(faces, selected_faces, fold_samples)
        else:
            fold_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        fold_ids, fold_basis, fold_target, fold_weights = samples_to_tensors(
            fold_samples,
            fold_sample_vertex_ids,
            vertices_local,
            strength=float(args.strength),
            max_abs_delta_rgb=float(args.max_abs_delta_rgb),
            sh_degree=int(args.sh_degree),
            device=device,
        )
        fold_proxy = evaluate_proxy(coeff, fold_ids, fold_basis, fold_target, fold_weights)
        row = {
            "group": int(group_idx),
            "view_names": [p.stem for p in fold_paths],
            "samples": int(fold_samples.count),
            "proxy": fold_proxy,
            "ids": fold_ids,
            "basis": fold_basis,
            "target": fold_target,
            "weights": fold_weights,
            "face_ids": fold_samples.face_ids,
        }
        cache.append(row)
        summary["fold_summaries"].append(
            {
                "group": int(group_idx),
                "view_count": int(len(fold_paths)),
                "view_names": row["view_names"][:16],
                "samples": int(fold_samples.count),
                "proxy": fold_proxy,
            }
        )
    return cache, summary


def carrier_holdout_certificate_for_faces(
    coeff: torch.Tensor,
    holdout_cache: list[dict[str, Any]],
    face_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    groups = int(args.patch_cert_carrier_holdout_groups)
    required_passing_raw = int(args.patch_cert_carrier_holdout_min_passing_groups)
    required_passing = max(groups, 0) if required_passing_raw <= 0 else required_passing_raw
    result: dict[str, Any] = {
        "enabled": bool(args.patch_cert_carrier_holdout_selector),
        "selection_unit": "patchcert_carrier",
        "test_usage": "none",
        "holdout_source": "policy_val_train_split",
        "grouping": str(args.patch_cert_carrier_holdout_grouping),
        "disjoint_from_policy_tuning": bool(args.patch_cert_carrier_holdout_disjoint),
        "groups": int(groups),
        "min_passing_groups": int(required_passing),
        "min_group_relative_gain": float(args.patch_cert_carrier_holdout_min_group_relative_gain),
        "min_group_samples": int(args.patch_cert_carrier_holdout_min_group_samples),
        "max_mse_regression": float(args.patch_cert_carrier_holdout_max_mse_regression),
        "passing_groups": 0,
        "eligible_groups": 0,
        "passed": False,
        "score": 0.0,
        "mean_relative_gain": 0.0,
        "min_relative_gain": 0.0,
        "cvar_loss": 0.0,
        "group_rows": [],
    }
    if not bool(args.patch_cert_carrier_holdout_selector):
        result["passed"] = True
        return result
    gains: list[float] = []
    losses: list[float] = []
    for row in holdout_cache:
        proxy = evaluate_proxy_for_faces(
            coeff,
            row["ids"],
            row["basis"],
            row["target"],
            row["weights"],
            row["face_ids"],
            face_ids,
        )
        samples = int(proxy.get("samples", 0))
        relative_gain = float(proxy.get("relative_gain", -1.0))
        mse_before = float(proxy.get("mse_before", 0.0))
        mse_after = float(proxy.get("mse_after", 0.0))
        mse_regression = float((mse_after - mse_before) / max(mse_before, 1e-12)) if samples > 0 else 0.0
        eligible = samples >= int(args.patch_cert_carrier_holdout_min_group_samples)
        gain_passed = relative_gain >= float(args.patch_cert_carrier_holdout_min_group_relative_gain)
        regression_passed = mse_regression <= float(args.patch_cert_carrier_holdout_max_mse_regression)
        passed = bool(eligible and gain_passed and regression_passed)
        if eligible:
            result["eligible_groups"] = int(result["eligible_groups"]) + 1
            gains.append(relative_gain)
            losses.append(max(-relative_gain, 0.0))
        if passed:
            result["passing_groups"] = int(result["passing_groups"]) + 1
        result["group_rows"].append(
            {
                "group": int(row.get("group", len(result["group_rows"]))),
                "samples": samples,
                "relative_gain": relative_gain,
                "mse_regression": mse_regression,
                "eligible": bool(eligible),
                "gain_passed": bool(gain_passed),
                "regression_passed": bool(regression_passed),
                "passed": bool(passed),
            }
        )
    if gains:
        gain_np = np.asarray(gains, dtype=np.float32)
        loss_np = np.asarray(losses, dtype=np.float32)
        cvar_count = max(int(math.ceil(float(args.patch_cert_carrier_holdout_cvar_fraction) * loss_np.size)), 1)
        cvar_loss = float(np.sort(loss_np)[-cvar_count:].mean())
        mean_gain = float(gain_np.mean())
        result["mean_relative_gain"] = mean_gain
        result["min_relative_gain"] = float(gain_np.min())
        result["cvar_loss"] = cvar_loss
        result["score"] = float(mean_gain - float(args.patch_cert_carrier_holdout_cvar_weight) * cvar_loss)
    result["passed"] = bool(int(result["passing_groups"]) >= required_passing)
    return result


def carrier_holdout_positive_tail_safety(cert: dict[str, Any]) -> dict[str, Any]:
    """Return the per-carrier risk check used by opt-in safe auto-prefix selection."""

    group_rows = cert.get("group_rows", [])
    eligible_gains = [
        float(row.get("relative_gain", -1.0))
        for row in group_rows
        if bool(row.get("eligible", False))
    ]
    score = float(cert.get("score", 0.0))
    mean_relative_gain = float(cert.get("mean_relative_gain", 0.0))
    min_relative_gain = float(min(eligible_gains)) if eligible_gains else float(cert.get("min_relative_gain", 0.0))
    cvar_loss = float(cert.get("cvar_loss", 0.0))
    passed = bool(cert.get("passed", False))
    positive = bool(score >= 0.0 and mean_relative_gain >= 0.0)
    tail_safe = bool(eligible_gains and min_relative_gain >= 0.0 and cvar_loss <= 1.0e-12)
    return {
        "passed": bool(passed and positive and tail_safe),
        "criterion": "individual_carrier_score_mean_min_holdout_gain_nonnegative",
        "carrier_passed": bool(passed),
        "positive": bool(positive),
        "tail_safe": bool(tail_safe),
        "score": score,
        "mean_relative_gain": mean_relative_gain,
        "min_relative_gain": min_relative_gain,
        "cvar_loss": cvar_loss,
        "eligible_groups": int(cert.get("eligible_groups", len(eligible_gains))),
    }


def select_holdout_stable_carriers(
    *,
    coeff: torch.Tensor,
    accepted_faces: list[int],
    patch_cert_by_face: dict[int, dict[str, Any]],
    holdout_cache: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any], dict[int, dict[str, Any]]]:
    summary: dict[str, Any] = {
        "enabled": bool(args.patch_cert_carrier_holdout_selector),
        "selection_unit": "whole_patchcert_carrier",
        "test_usage": "none",
        "holdout_source": "policy_val_train_split",
        "grouping": str(args.patch_cert_carrier_holdout_grouping),
        "disjoint_from_policy_tuning": bool(args.patch_cert_carrier_holdout_disjoint),
        "groups": int(args.patch_cert_carrier_holdout_groups),
        "min_passing_groups": (
            int(args.patch_cert_carrier_holdout_groups)
            if int(args.patch_cert_carrier_holdout_min_passing_groups) <= 0
            else int(args.patch_cert_carrier_holdout_min_passing_groups)
        ),
        "min_group_relative_gain": float(args.patch_cert_carrier_holdout_min_group_relative_gain),
        "min_group_samples": int(args.patch_cert_carrier_holdout_min_group_samples),
        "max_mse_regression": float(args.patch_cert_carrier_holdout_max_mse_regression),
        "cvar_fraction": float(args.patch_cert_carrier_holdout_cvar_fraction),
        "cvar_weight": float(args.patch_cert_carrier_holdout_cvar_weight),
        "max_carriers": int(args.patch_cert_carrier_holdout_max_carriers),
        "auto_prefix": bool(args.patch_cert_carrier_holdout_auto_prefix),
        "auto_prefix_min_faces": int(args.patch_cert_carrier_holdout_auto_prefix_min_faces),
        "auto_prefix_effective_min_faces": int(args.patch_cert_carrier_holdout_auto_prefix_min_faces),
        "auto_prefix_face_bonus": float(args.patch_cert_carrier_holdout_auto_prefix_face_bonus),
        "auto_prefix_positive_tail_safe": bool(args.patch_cert_carrier_holdout_auto_prefix_positive_tail_safe),
        "auto_prefix_min_faces_relaxed_by_tail_safety": False,
        "auto_prefix_tail_safety_stop": {},
        "auto_prefix_certificate": {},
        "auto_prefix_candidates": [],
        "input_faces": int(len(accepted_faces)),
        "input_carriers": 0,
        "selected_faces": int(len(accepted_faces)),
        "selected_carriers": 0,
        "rejected_carriers": 0,
        "blocked_reason": None,
        "carrier_rows": [],
    }
    if not bool(args.patch_cert_carrier_holdout_selector):
        return list(accepted_faces), summary, {}
    if not accepted_faces:
        summary["selected_faces"] = 0
        return [], summary, {}
    if not holdout_cache:
        summary["blocked_reason"] = "empty_holdout_cache"
        summary["selected_faces"] = 0
        summary["rejected_carriers"] = 0
        return [], summary, {}

    accepted_set = {int(fid) for fid in accepted_faces}
    carriers: dict[str, dict[str, Any]] = {}
    for fid in accepted_faces:
        face_id = int(fid)
        patch = patch_cert_by_face.get(face_id, {})
        carrier_id = patch_carrier_id(patch, face_id)
        carrier_faces = sorted({int(x) for x in patch_carrier_faces(patch, face_id)})
        if not carrier_faces:
            carrier_faces = [face_id]
        row = carriers.setdefault(
            carrier_id,
            {
                "carrier_id": carrier_id,
                "seed_face": patch_seed_face(patch, face_id),
                "faces": sorted(set(carrier_faces)),
                "observed_accepted_faces": [],
            },
        )
        row["faces"] = sorted(set([int(x) for x in row["faces"]] + carrier_faces))
        row["observed_accepted_faces"] = sorted(set([int(x) for x in row["observed_accepted_faces"]] + [face_id]))

    rows: list[dict[str, Any]] = []
    for carrier_id, carrier in sorted(carriers.items()):
        faces = sorted({int(fid) for fid in carrier.get("faces", [])})
        missing_faces = [fid for fid in faces if fid not in accepted_set]
        split_rejected = bool(missing_faces)
        if split_rejected:
            cert = {
                "enabled": bool(args.patch_cert_carrier_holdout_selector),
                "selection_unit": "patchcert_carrier",
                "test_usage": "none",
                "holdout_source": "policy_val_train_split",
                "grouping": str(args.patch_cert_carrier_holdout_grouping),
                "disjoint_from_policy_tuning": bool(args.patch_cert_carrier_holdout_disjoint),
                "groups": int(args.patch_cert_carrier_holdout_groups),
                "min_passing_groups": int(summary["min_passing_groups"]),
                "passed": False,
                "blocked_reason": "carrier_split_before_holdout",
                "missing_accepted_faces": missing_faces[:20],
                "missing_accepted_face_count": int(len(missing_faces)),
            }
        else:
            cert = carrier_holdout_certificate_for_faces(coeff, holdout_cache, faces, args)
        tail_safety = carrier_holdout_positive_tail_safety(cert)
        rows.append(
            {
                "carrier_id": str(carrier_id),
                "seed_face": int(carrier.get("seed_face", faces[0] if faces else -1)),
                "faces": faces,
                "observed_accepted_faces": [int(fid) for fid in carrier.get("observed_accepted_faces", [])],
                "face_count": int(len(faces)),
                "certificate": cert,
                "passed": bool(cert.get("passed", False)) and not split_rejected,
                "split_rejected": bool(split_rejected),
                "missing_accepted_face_count": int(len(missing_faces)),
                "score": float(cert.get("score", 0.0)),
                "mean_relative_gain": float(cert.get("mean_relative_gain", 0.0)),
                "cvar_loss": float(cert.get("cvar_loss", 0.0)),
                "positive_tail_safety": tail_safety,
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("passed", False)),
            float(row.get("score", 0.0)),
            float(row.get("mean_relative_gain", 0.0)),
            int(row.get("face_count", 0)),
        ),
        reverse=True,
    )
    selected_rows: list[dict[str, Any]] = []
    selected_faces: list[int] = []
    max_carriers = int(args.patch_cert_carrier_holdout_max_carriers)
    max_faces = max(int(args.max_faces_to_apply), 0)
    if bool(args.patch_cert_carrier_holdout_auto_prefix):
        prefix_rows: list[dict[str, Any]] = []
        prefix_faces: list[int] = []
        prefix_face_set: set[int] = set()
        best_rows: list[dict[str, Any]] = []
        best_faces: list[int] = []
        best_cert: dict[str, Any] = {}
        best_key: tuple[float, float, int, int] | None = None
        best_under_floor_rows: list[dict[str, Any]] = []
        best_under_floor_faces: list[int] = []
        best_under_floor_cert: dict[str, Any] = {}
        best_under_floor_key: tuple[float, float, int, int] | None = None
        stopped_by_tail_safety = False
        min_prefix_faces = int(args.patch_cert_carrier_holdout_auto_prefix_min_faces)
        if min_prefix_faces > 0 and int(summary["input_faces"]) > 0:
            min_prefix_faces = min(min_prefix_faces, int(summary["input_faces"]))
        summary["auto_prefix_effective_min_faces"] = int(min_prefix_faces)
        face_bonus_weight = float(args.patch_cert_carrier_holdout_auto_prefix_face_bonus)
        coverage_aware = bool(min_prefix_faces > 0 or face_bonus_weight > 0.0)
        require_positive_tail_safe = bool(args.patch_cert_carrier_holdout_auto_prefix_positive_tail_safe)
        for row in rows:
            if not bool(row.get("passed", False)):
                continue
            tail_safety = row.get("positive_tail_safety", {})
            if require_positive_tail_safe and not bool(tail_safety.get("passed", False)):
                stopped_by_tail_safety = True
                summary["auto_prefix_candidates"].append(
                    {
                        "prefix_carriers": int(len(prefix_rows)),
                        "prefix_faces": int(len(prefix_faces)),
                        "last_carrier_id": str(row.get("carrier_id", "")),
                        "last_carrier_positive_tail_safety": tail_safety,
                        "passed": False,
                        "selected_into_prefix": False,
                        "blocked_by_positive_tail_safety": True,
                        "positive_tail_safe_required": True,
                    }
                )
                summary["auto_prefix_tail_safety_stop"] = {
                    "blocked_carrier_id": str(row.get("carrier_id", "")),
                    "blocked_faces": [int(fid) for fid in row.get("faces", [])],
                    "current_prefix_faces": int(len(prefix_faces)),
                    "current_prefix_carriers": int(len(prefix_rows)),
                    "risk": tail_safety,
                }
                break
            if max_carriers > 0 and len(prefix_rows) >= max_carriers:
                continue
            faces = [int(fid) for fid in row.get("faces", [])]
            new_faces = [fid for fid in faces if fid not in prefix_face_set]
            if max_faces > 0 and len(prefix_faces) + len(new_faces) > max_faces:
                continue
            prefix_rows.append(row)
            prefix_faces.extend(new_faces)
            prefix_face_set.update(new_faces)
            cumulative_cert = carrier_holdout_certificate_for_faces(coeff, holdout_cache, prefix_faces, args)
            prefix_face_count = int(len(prefix_faces))
            prefix_row_count = int(len(prefix_rows))
            holdout_score = float(cumulative_cert.get("score", 0.0))
            mean_relative_gain = float(cumulative_cert.get("mean_relative_gain", 0.0))
            face_bonus = float(face_bonus_weight * math.log1p(max(prefix_face_count, 0)))
            coverage_score = float(holdout_score + face_bonus)
            coverage_floor_passed = bool(prefix_face_count >= min_prefix_faces)
            candidate = {
                "prefix_carriers": prefix_row_count,
                "prefix_faces": prefix_face_count,
                "last_carrier_id": str(row.get("carrier_id", "")),
                "last_carrier_positive_tail_safety": row.get("positive_tail_safety", {}),
                "passed": bool(cumulative_cert.get("passed", False)),
                "score": holdout_score,
                "mean_relative_gain": mean_relative_gain,
                "cvar_loss": float(cumulative_cert.get("cvar_loss", 0.0)),
                "coverage_floor_passed": coverage_floor_passed,
                "positive_tail_safe_required": bool(require_positive_tail_safe),
                "selected_into_prefix": True,
                "blocked_by_positive_tail_safety": False,
                "coverage_bonus": face_bonus,
                "coverage_score": coverage_score,
            }
            summary["auto_prefix_candidates"].append(candidate)
            if coverage_aware:
                key = (
                    coverage_score,
                    mean_relative_gain,
                    prefix_face_count,
                    -prefix_row_count,
                )
            else:
                key = (
                    holdout_score,
                    mean_relative_gain,
                    -prefix_face_count,
                    -prefix_row_count,
                )
            if (
                bool(cumulative_cert.get("passed", False))
                and holdout_score >= 0.0
                and coverage_floor_passed
            ):
                if best_key is None or key > best_key:
                    best_key = key
                    best_rows = list(prefix_rows)
                    best_faces = list(prefix_faces)
                    best_cert = cumulative_cert
            elif (
                require_positive_tail_safe
                and bool(cumulative_cert.get("passed", False))
                and holdout_score >= 0.0
            ):
                if best_under_floor_key is None or key > best_under_floor_key:
                    best_under_floor_key = key
                    best_under_floor_rows = list(prefix_rows)
                    best_under_floor_faces = list(prefix_faces)
                    best_under_floor_cert = cumulative_cert
        if require_positive_tail_safe and not best_rows and stopped_by_tail_safety and best_under_floor_rows:
            best_rows = list(best_under_floor_rows)
            best_faces = list(best_under_floor_faces)
            best_cert = best_under_floor_cert
            best_cert["coverage_floor_passed"] = False
            best_cert["coverage_floor_relaxed_by_tail_safety"] = True
            best_cert["requested_min_faces"] = int(args.patch_cert_carrier_holdout_auto_prefix_min_faces)
            best_cert["effective_min_faces_before_tail_safety"] = int(min_prefix_faces)
            summary["auto_prefix_min_faces_relaxed_by_tail_safety"] = True
            summary["auto_prefix_effective_min_faces"] = int(len(best_faces))
        selected_rows = best_rows
        selected_faces = best_faces
        summary["auto_prefix_certificate"] = best_cert
        if not selected_rows:
            if require_positive_tail_safe and stopped_by_tail_safety:
                summary["blocked_reason"] = "auto_prefix_stopped_by_positive_tail_safety_before_selectable_prefix"
            elif min_prefix_faces > 0 and any(bool(row.get("passed", False)) for row in rows):
                summary["blocked_reason"] = "auto_prefix_no_prefix_met_coverage_floor"
            else:
                summary["blocked_reason"] = "auto_prefix_no_nonnegative_cumulative_holdout_score"
    else:
        for row in rows:
            if not bool(row.get("passed", False)):
                continue
            if max_carriers > 0 and len(selected_rows) >= max_carriers:
                continue
            faces = [int(fid) for fid in row.get("faces", [])]
            if max_faces > 0 and len(selected_faces) + len(faces) > max_faces:
                continue
            selected_rows.append(row)
            selected_faces.extend(faces)

    by_face: dict[int, dict[str, Any]] = {}
    selected_face_set = {int(fid) for fid in selected_faces}
    for row in rows:
        row["selected"] = bool(any(int(fid) in selected_face_set for fid in row.get("faces", [])))
        for fid in row.get("faces", []):
            by_face[int(fid)] = row

    summary["input_carriers"] = int(len(rows))
    summary["selected_carriers"] = int(len(selected_rows))
    summary["selected_faces"] = int(len(selected_faces))
    summary["rejected_carriers"] = int(len(rows) - len(selected_rows))
    summary["carrier_rows"] = rows[:50]
    selected_order = [fid for fid in accepted_faces if int(fid) in selected_face_set]
    return selected_order, summary, by_face


def apply_patch_cert_seed_rescue(
    *,
    strict_face_candidates: list[int],
    selected_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    face_policy: dict[int, dict[str, float]],
    face_view_gain_certificate: dict[int, dict[str, Any]],
    face_view_gain_certificate_summary: dict[str, Any],
    crossfold_face_gain: dict[int, dict[str, Any]],
    crossfold_face_gain_summary: dict[str, Any],
    face_view_consensus: dict[int, dict[str, Any]],
    face_view_consensus_summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    """Add deterministic group-first PatchCert seeds without weakening final gates."""

    enabled_aux = [
        (
            "face_view_gain",
            bool(face_view_gain_certificate_summary.get("enabled", False)),
            face_view_gain_certificate,
        ),
        (
            "face_crossfold_gain",
            bool(crossfold_face_gain_summary.get("enabled", False)),
            crossfold_face_gain,
        ),
        (
            "face_view_consensus",
            bool(face_view_consensus_summary.get("enabled", False)),
            face_view_consensus,
        ),
    ]
    enabled_aux = [(name, cert_map) for name, enabled, cert_map in enabled_aux if enabled]
    required_witnesses = min(
        int(args.patch_cert_seed_rescue_min_aux_witnesses),
        len(enabled_aux),
    )
    summary: dict[str, Any] = {
        "enabled": bool(args.patch_cert_seed_rescue),
        "mode": "group_first_policy_val_seed_rescue",
        "trigger_min_candidates": int(args.patch_cert_seed_rescue_min_candidates),
        "max_seeds": int(args.patch_cert_seed_rescue_max_seeds),
        "min_aux_witnesses": int(args.patch_cert_seed_rescue_min_aux_witnesses),
        "effective_required_aux_witnesses": int(required_witnesses),
        "enabled_aux_witnesses": [name for name, _ in enabled_aux],
        "base_candidate_count": int(len(strict_face_candidates)),
        "triggered": False,
        "policy_eligible_faces": 0,
        "aux_witness_eligible_faces": 0,
        "added_seed_count": 0,
        "added_seed_faces": [],
        "witness_histogram": {},
    }
    if (
        not bool(args.patch_cert_seed_rescue)
        or int(args.patch_cert_rings) <= 0
        or len(strict_face_candidates) >= int(args.patch_cert_seed_rescue_min_candidates)
        or int(args.patch_cert_seed_rescue_max_seeds) <= 0
    ):
        return list(strict_face_candidates), summary

    summary["triggered"] = True
    strict_set = {int(fid) for fid in strict_face_candidates}
    scored: list[tuple[tuple[float, float, float, float, float], int, int]] = []
    witness_hist: dict[int, int] = {}
    for fid_raw in selected_faces:
        fid = int(fid_raw)
        if fid in strict_set:
            continue
        policy = face_policy.get(fid, {})
        samples = int(policy.get("samples", 0))
        relative_gain = float(policy.get("relative_gain", -1.0))
        if samples < int(args.min_face_policy_val_samples):
            continue
        if relative_gain < float(args.min_face_policy_val_relative_gain):
            continue
        summary["policy_eligible_faces"] = int(summary["policy_eligible_faces"]) + 1
        witness_count = 0
        for _, cert_map in enabled_aux:
            if bool(cert_map.get(fid, {}).get("passed", False)):
                witness_count += 1
        witness_hist[witness_count] = int(witness_hist.get(witness_count, 0)) + 1
        if witness_count < required_witnesses:
            continue
        summary["aux_witness_eligible_faces"] = int(summary["aux_witness_eligible_faces"]) + 1
        stats = face_stats.get(fid, {})
        score = (
            float(witness_count),
            float(relative_gain),
            float(samples),
            float(stats.get("score", 0.0)),
            float(stats.get("pixel_count", 0.0)),
        )
        scored.append((score, fid, witness_count))

    scored.sort(key=lambda row: row[0], reverse=True)
    added = [fid for _, fid, _ in scored[: int(args.patch_cert_seed_rescue_max_seeds)]]
    summary["added_seed_count"] = int(len(added))
    summary["added_seed_faces"] = [int(fid) for fid in added[:50]]
    summary["witness_histogram"] = {str(k): int(v) for k, v in sorted(witness_hist.items())}
    return list(strict_face_candidates) + added, summary


def clone_face_coeffs(coeff: torch.Tensor, selected_faces: list[int], face_ids: list[int]) -> dict[int, torch.Tensor]:
    if not face_ids or coeff.numel() == 0:
        return {}
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    out: dict[int, torch.Tensor] = {}
    for fid in face_ids:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        out[int(fid)] = coeff[row * 3 : row * 3 + 3].clone()
    return out


def restore_face_coeffs(coeff: torch.Tensor, selected_faces: list[int], snapshot: dict[int, torch.Tensor]) -> None:
    if not snapshot or coeff.numel() == 0:
        return
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    for fid, value in snapshot.items():
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        coeff[row * 3 : row * 3 + 3] = value.to(device=coeff.device, dtype=coeff.dtype)


def fit_patch_scale(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
    face_ids: list[int],
    min_samples: int,
) -> tuple[float, int, float]:
    if not face_ids or sample_vertex_ids.numel() == 0:
        return 0.0, 0, 0.0
    mask = np.isin(sample_face_ids.astype(np.int64, copy=False), np.asarray(face_ids, dtype=np.int64))
    sample_count = int(mask.sum())
    if sample_count < max(int(min_samples), 1):
        return 0.0, sample_count, 0.0
    idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long, device=sample_vertex_ids.device)
    with torch.no_grad():
        pred = _predict(coeff, sample_vertex_ids[idx], weighted_basis[idx])
        y = target[idx]
        w = weights[idx].clamp_min(1e-8).view(-1, 1)
        numerator = float((w * pred * y).sum().detach().cpu().item())
        denominator = float((w * pred * pred).sum().detach().cpu().item())
    if denominator <= 1e-12:
        return 0.0, sample_count, 0.0
    raw_scale = numerator / denominator
    return float(min(max(raw_scale, 0.0), 1.0)), sample_count, float(raw_scale)


def scale_face_coeffs(
    coeff: torch.Tensor,
    selected_faces: list[int],
    face_ids: list[int],
    scale: float,
) -> None:
    if not face_ids or coeff.numel() == 0:
        return
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    for fid in face_ids:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        coeff[row * 3 : row * 3 + 3] = coeff[row * 3 : row * 3 + 3] * float(scale)


def fit_patch_cluster_shared_basis(
    coeff: torch.Tensor,
    selected_faces: list[int],
    face_ids: list[int],
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_face_ids: np.ndarray,
    sample_view_names: list[str] | None,
    args: argparse.Namespace,
    faces: torch.Tensor | None = None,
    vertices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Refit a low-rank three-corner residual basis for a certified patch.

    Existing face-local fitting gives every face its own three local vertex
    residuals.  This routine constrains all faces in a PatchCert carrier to a
    shared corner-indexed SH residual basis, optionally with one learned
    positive scale per face.  It writes the carrier coefficients only if the
    fixed train-fit regression bound is satisfied.  The later policy-val and
    fold certificates evaluate the materialized coefficients.
    """

    enabled = bool(args.patch_cert_cluster_basis)
    mode = str(args.patch_cert_cluster_basis_mode)
    basis_type = {
        "scaled": "scaled_shared_patch_corner_sh",
        "rank2": "rank2_mixture_patch_corner_sh",
        "chart_linear": "chart_linear_patch_corner_sh",
        "chart_quad": "chart_quadratic_patch_corner_sh",
        "field_linear": "view_consistent_linear_residual_field",
        "field_quad": "view_consistent_quadratic_residual_field",
    }.get(mode, "shared_patch_corner_sh")
    result: dict[str, Any] = {
        "enabled": enabled,
        "basis_type": basis_type,
        "mode": mode,
        "rank": 2
        if mode == "rank2"
        else (6 if mode in {"chart_quad", "field_quad"} else (3 if mode in {"chart_linear", "field_linear"} else 1)),
        "faces": [int(fid) for fid in face_ids],
        "patch_size": int(len(face_ids)),
        "steps": int(args.patch_cert_cluster_basis_steps),
        "lr": float(args.patch_cert_cluster_basis_lr),
        "min_samples": int(args.patch_cert_cluster_basis_min_samples),
        "max_scale": float(args.patch_cert_cluster_basis_max_scale),
        "max_fit_mse_regression": float(args.patch_cert_cluster_basis_max_fit_mse_regression),
        "init": str(args.patch_cert_cluster_basis_init),
        "view_hinge_weight": float(args.patch_cert_cluster_basis_view_hinge_weight),
        "view_hinge_min_samples": int(args.patch_cert_cluster_basis_view_hinge_min_samples),
        "geometry_smooth_weight": float(args.patch_cert_cluster_basis_geometry_smooth_weight),
        "samples": 0,
        "applied": False,
        "passed": not enabled,
    }
    if not enabled:
        return result
    if len(face_ids) <= 1:
        result.update({"passed": True, "skip_reason": "single_face_patch"})
        return result
    if sample_vertex_ids.numel() == 0 or coeff.numel() == 0:
        result.update({"passed": False, "rejected_reason": "no_fit_samples"})
        return result

    mask = np.isin(sample_face_ids.astype(np.int64, copy=False), np.asarray(face_ids, dtype=np.int64))
    sample_count = int(mask.sum())
    result["samples"] = sample_count
    if sample_count < max(int(args.patch_cert_cluster_basis_min_samples), 1):
        result.update({"passed": False, "rejected_reason": "insufficient_fit_samples"})
        return result

    masked_face_ids = sample_face_ids.astype(np.int64, copy=False)[mask]
    face_to_patch_row = {int(fid): i for i, fid in enumerate(face_ids)}
    patch_sample_rows = torch.as_tensor(
        [face_to_patch_row[int(fid)] for fid in masked_face_ids],
        dtype=torch.long,
        device=coeff.device,
    )
    idx = torch.as_tensor(np.nonzero(mask)[0], dtype=torch.long, device=sample_vertex_ids.device)
    ids = sample_vertex_ids[idx].to(device=coeff.device)
    basis = weighted_basis[idx].to(device=coeff.device)
    y = target[idx].to(device=coeff.device)
    w = weights[idx].to(device=coeff.device).clamp_min(1e-8)
    corner_ids = torch.remainder(ids, 3).long()
    basis_count = int(basis.shape[2]) if basis.ndim == 3 else int(coeff.shape[1])

    face_to_selected = {int(fid): row for row, fid in enumerate(selected_faces)}
    local_rows_by_face: dict[int, int] = {}
    missing_patch_faces: list[int] = []
    for fid in face_ids:
        row = face_to_selected.get(int(fid))
        if row is None:
            missing_patch_faces.append(int(fid))
        else:
            local_rows_by_face[int(fid)] = int(row)
    if missing_patch_faces:
        result.update(
            {
                "passed": False,
                "rejected_reason": "patch_faces_not_in_selected_set",
                "missing_patch_faces": missing_patch_faces[:20],
            }
        )
        return result
    local_rows = [local_rows_by_face[int(fid)] for fid in face_ids]

    max_abs_dc_coeff = float(args.max_abs_delta_rgb) / float(C0)
    max_abs_sh_coeff = (
        float(args.max_abs_sh_coeff)
        if float(args.max_abs_sh_coeff) > 0
        else float(args.max_abs_delta_rgb) / float(C1)
    )
    bounds = torch.full((basis_count,), float(max_abs_sh_coeff), dtype=torch.float32, device=coeff.device)
    bounds[0] = float(max_abs_dc_coeff)
    bounds = bounds.view(1, basis_count, 1).clamp_min(1e-12)

    def predict_shared(shared: torch.Tensor, face_scale: torch.Tensor | None = None) -> torch.Tensor:
        sample_coeff = shared[corner_ids]
        if face_scale is not None:
            scale_shape = (-1,) + (1,) * (sample_coeff.ndim - 1)
            sample_coeff = sample_coeff * face_scale[patch_sample_rows].view(*scale_shape)
            sample_coeff = torch.clamp(sample_coeff, min=-bounds, max=bounds)
        return (sample_coeff * basis[:, :, :, None]).sum(dim=(1, 2))

    def predict_rank2(components: torch.Tensor, face_mix: torch.Tensor) -> torch.Tensor:
        components_by_corner = components.permute(1, 0, 2, 3)
        sample_components = components_by_corner[corner_ids]
        sample_mix = face_mix[patch_sample_rows].view(-1, 1, int(face_mix.shape[1]), 1, 1)
        sample_coeff = (sample_components * sample_mix).sum(dim=2)
        return (sample_coeff * basis[:, :, :, None]).sum(dim=(1, 2))

    chart_features: torch.Tensor | None = None
    chart_uv_scale = 0.0
    chart_svd_scale = 0.0
    patch_vertex_ids: torch.Tensor | None = None
    if mode in {"chart_linear", "chart_quad", "field_linear", "field_quad"}:
        if faces is None or vertices is None:
            result.update({"passed": False, "rejected_reason": "chart_geometry_unavailable"})
            return result
        face_index = torch.as_tensor(face_ids, dtype=torch.long, device=faces.device)
        patch_vertex_ids = faces[face_index].long().to(device=vertices.device)
        patch_points = vertices[patch_vertex_ids].float().to(device=coeff.device)
        flat_points = patch_points.reshape(-1, 3)
        center = flat_points.mean(dim=0, keepdim=True)
        centered = flat_points - center
        try:
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            axes = vh[:2].T
            chart_svd_scale = float(singular_values[:2].max().detach().cpu().item())
        except RuntimeError:
            axes = torch.eye(3, 2, dtype=torch.float32, device=coeff.device)
            chart_svd_scale = 0.0
        uv = (patch_points.reshape(-1, 3) - center) @ axes
        uv_scale = uv.norm(dim=1).max().clamp_min(1e-6)
        uv = (uv / uv_scale).reshape(len(face_ids), 3, 2)
        ones = torch.ones((len(face_ids), 3, 1), dtype=torch.float32, device=coeff.device)
        if mode in {"chart_quad", "field_quad"}:
            u = uv[..., 0:1]
            v = uv[..., 1:2]
            chart_features = torch.cat([ones, uv, u * u, u * v, v * v], dim=-1)
        else:
            chart_features = torch.cat([ones, uv], dim=-1)
        chart_uv_scale = float(uv_scale.detach().cpu().item())

    def gather_chart_sample_features(features: torch.Tensor) -> torch.Tensor:
        sample_face_features = features[patch_sample_rows]
        gather_idx = corner_ids[..., None].expand(-1, -1, int(features.shape[-1]))
        return torch.gather(sample_face_features, 1, gather_idx)

    chart_sample_features = gather_chart_sample_features(chart_features) if chart_features is not None else None

    def predict_chart_linear(params: torch.Tensor) -> torch.Tensor:
        assert chart_sample_features is not None
        sample_coeff = torch.einsum("ncf,fbr->ncbr", chart_sample_features, params)
        sample_coeff = torch.clamp(sample_coeff, min=-bounds, max=bounds)
        return (sample_coeff * basis[:, :, :, None]).sum(dim=(1, 2))

    with torch.no_grad():
        independent_before = _weighted_mse(coeff, ids, basis, y, w)
        zero = torch.zeros((3, basis_count, 3), dtype=torch.float32, device=coeff.device)
        zero_pred = (
            predict_chart_linear(torch.zeros((int(chart_features.shape[-1]), basis_count, 3), dtype=torch.float32, device=coeff.device))
            if chart_features is not None
            else predict_shared(zero)
        )
        zero_mse = (((zero_pred - y) ** 2) * w[:, None]).sum() / (w.sum().clamp_min(1e-8) * 3.0)

    rows = [coeff[row * 3 : row * 3 + 3, :basis_count, :].detach() for row in local_rows]
    face_coeff_tensor = (
        torch.stack(rows, dim=0).to(device=coeff.device, dtype=torch.float32)
        if rows
        else torch.zeros((0, 3, basis_count, 3), dtype=torch.float32, device=coeff.device)
    )
    init = torch.zeros((3, basis_count, 3), dtype=torch.float32, device=coeff.device)
    if str(args.patch_cert_cluster_basis_init) == "mean" and face_coeff_tensor.numel():
        init = face_coeff_tensor.mean(dim=0)
    max_scale = max(float(args.patch_cert_cluster_basis_max_scale), 1e-6)
    final_data = zero_mse
    final_mag = torch.zeros((), dtype=torch.float32, device=coeff.device)
    final_sh_mag = torch.zeros((), dtype=torch.float32, device=coeff.device)
    final_scale_reg = torch.zeros((), dtype=torch.float32, device=coeff.device)

    shared: torch.Tensor | None = None
    face_scale: torch.Tensor | None = None
    components: torch.Tensor | None = None
    face_mix: torch.Tensor | None = None
    chart_params: torch.Tensor | None = None
    if mode in {"chart_linear", "chart_quad", "field_linear", "field_quad"}:
        assert chart_features is not None
        param_init = torch.zeros((int(chart_features.shape[-1]), basis_count, 3), dtype=torch.float32, device=coeff.device)
        if str(args.patch_cert_cluster_basis_init) == "mean":
            param_init[0] = init.mean(dim=0)
        param_init = torch.clamp(param_init / bounds, -0.95, 0.95)
        param = torch.atanh(param_init).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([param], lr=float(args.patch_cert_cluster_basis_lr))
        for _ in range(max(int(args.patch_cert_cluster_basis_steps), 0)):
            current_params = bounds * torch.tanh(param)
            pred = predict_chart_linear(current_params)
            data = (((pred - y) ** 2) * w[:, None]).sum() / (w.sum().clamp_min(1e-8) * 3.0)
            mag = (current_params[0, 0, :] ** 2).mean()
            sh_mag = (
                (current_params[:, 1:, :] ** 2).mean()
                if basis_count > 1
                else torch.zeros((), dtype=torch.float32, device=coeff.device)
            )
            linear_reg = (current_params[1:] ** 2).mean()
            field_geometry_smooth = torch.zeros((), dtype=torch.float32, device=coeff.device)
            if mode in {"field_linear", "field_quad"} and patch_vertex_ids is not None:
                corner_coeff = torch.einsum("mcf,fbr->mcbr", chart_features, current_params)
                field_geometry_smooth = _duplicate_source_smooth_loss(
                    corner_coeff.reshape(-1, basis_count, 3),
                    patch_vertex_ids.to(device=coeff.device).reshape(-1),
                )
            loss = (
                data
                + float(args.lambda_mag) * mag
                + float(args.lambda_sh1_mag) * sh_mag
                + 0.001 * linear_reg
                + float(args.patch_cert_cluster_basis_geometry_smooth_weight) * field_geometry_smooth
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_data = data.detach()
            final_mag = mag.detach()
            final_sh_mag = sh_mag.detach()
            final_scale_reg = (linear_reg + field_geometry_smooth).detach()
        with torch.no_grad():
            chart_params = (bounds * torch.tanh(param)).detach()
            final_mse = (((predict_chart_linear(chart_params) - y) ** 2) * w[:, None]).sum() / (
                w.sum().clamp_min(1e-8) * 3.0
            )
    elif mode == "rank2":
        component_bounds = bounds.view(1, 1, basis_count, 1)
        component_init = torch.zeros((2, 3, basis_count, 3), dtype=torch.float32, device=coeff.device)
        if str(args.patch_cert_cluster_basis_init) == "mean" and face_coeff_tensor.numel():
            mean = face_coeff_tensor.mean(dim=0)
            if int(face_coeff_tensor.shape[0]) > 1:
                centered = face_coeff_tensor - mean.view(1, 3, basis_count, 3)
                norms = centered.reshape(int(centered.shape[0]), -1).norm(dim=1)
                direction = centered[int(torch.argmax(norms).detach().cpu().item())]
                component_init[0] = torch.clamp(mean + 0.5 * direction, min=-bounds, max=bounds)
                component_init[1] = torch.clamp(mean - 0.5 * direction, min=-bounds, max=bounds)
            else:
                component_init[0] = mean
                component_init[1] = mean
        component_param_init = torch.clamp(component_init / component_bounds, -0.95, 0.95)
        component_param = torch.atanh(component_param_init).detach().clone().requires_grad_(True)
        mix_logits = torch.zeros((len(face_ids), 2), dtype=torch.float32, device=coeff.device)
        if str(args.patch_cert_cluster_basis_init) == "mean" and int(face_coeff_tensor.shape[0]) > 1:
            with torch.no_grad():
                d0 = ((face_coeff_tensor - component_init[0].view(1, 3, basis_count, 3)) ** 2).mean(dim=(1, 2, 3))
                d1 = ((face_coeff_tensor - component_init[1].view(1, 3, basis_count, 3)) ** 2).mean(dim=(1, 2, 3))
                scale = torch.stack([d0, d1]).median().clamp_min(1e-8)
                mix_logits[:, 0] = -d0 / scale
                mix_logits[:, 1] = -d1 / scale
        mix_logits = mix_logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([component_param, mix_logits], lr=float(args.patch_cert_cluster_basis_lr))
        for _ in range(max(int(args.patch_cert_cluster_basis_steps), 0)):
            current_components = component_bounds * torch.tanh(component_param)
            current_mix = torch.softmax(mix_logits, dim=-1)
            pred = predict_rank2(current_components, current_mix)
            data = (((pred - y) ** 2) * w[:, None]).sum() / (w.sum().clamp_min(1e-8) * 3.0)
            mag = (current_components[:, :, 0, :] ** 2).mean()
            sh_mag = (
                (current_components[:, :, 1:, :] ** 2).mean()
                if basis_count > 1
                else torch.zeros((), dtype=torch.float32, device=coeff.device)
            )
            mix_reg = ((current_mix - 0.5) ** 2).mean()
            loss = data + float(args.lambda_mag) * mag + float(args.lambda_sh1_mag) * sh_mag + 0.0005 * mix_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_data = data.detach()
            final_mag = mag.detach()
            final_sh_mag = sh_mag.detach()
            final_scale_reg = mix_reg.detach()
        with torch.no_grad():
            components = (component_bounds * torch.tanh(component_param)).detach()
            face_mix = torch.softmax(mix_logits, dim=-1).detach()
            final_mse = (((predict_rank2(components, face_mix) - y) ** 2) * w[:, None]).sum() / (
                w.sum().clamp_min(1e-8) * 3.0
            )
    else:
        param_init = torch.clamp(init / bounds, -0.95, 0.95)
        param = torch.atanh(param_init).detach().clone().requires_grad_(True)
        optim_params: list[torch.Tensor] = [param]
        scale_param: torch.Tensor | None = None
        if mode == "scaled":
            init_scale = min(max(1.0 / max_scale, 1e-4), 1.0 - 1e-4)
            scale_init = torch.full((len(face_ids),), init_scale, dtype=torch.float32, device=coeff.device)
            scale_param = torch.logit(scale_init).detach().clone().requires_grad_(True)
            optim_params.append(scale_param)
        optimizer = torch.optim.Adam(optim_params, lr=float(args.patch_cert_cluster_basis_lr))
        for _ in range(max(int(args.patch_cert_cluster_basis_steps), 0)):
            current_shared = bounds * torch.tanh(param)
            current_scale = max_scale * torch.sigmoid(scale_param) if scale_param is not None else None
            pred = predict_shared(current_shared, current_scale)
            data = (((pred - y) ** 2) * w[:, None]).sum() / (w.sum().clamp_min(1e-8) * 3.0)
            mag = (current_shared[:, 0, :] ** 2).mean()
            sh_mag = (
                (current_shared[:, 1:, :] ** 2).mean()
                if basis_count > 1
                else torch.zeros((), dtype=torch.float32, device=coeff.device)
            )
            scale_reg = (
                ((current_scale - 1.0) ** 2).mean()
                if current_scale is not None
                else torch.zeros((), dtype=torch.float32, device=coeff.device)
            )
            loss = data + float(args.lambda_mag) * mag + float(args.lambda_sh1_mag) * sh_mag + 0.001 * scale_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_data = data.detach()
            final_mag = mag.detach()
            final_sh_mag = sh_mag.detach()
            final_scale_reg = scale_reg.detach()
        with torch.no_grad():
            shared = (bounds * torch.tanh(param)).detach()
            face_scale = (max_scale * torch.sigmoid(scale_param)).detach() if scale_param is not None else None
            final_mse = (((predict_shared(shared, face_scale) - y) ** 2) * w[:, None]).sum() / (
                w.sum().clamp_min(1e-8) * 3.0
            )

    independent_value = float(independent_before.detach().cpu().item())
    final_value = float(final_mse.detach().cpu().item())
    zero_value = float(zero_mse.detach().cpu().item())
    regression = (final_value - independent_value) / max(independent_value, 1e-12)
    relative_gain = (zero_value - final_value) / max(zero_value, 1e-12)
    passed = bool(regression <= float(args.patch_cert_cluster_basis_max_fit_mse_regression))

    result.update(
        {
            "independent_fit_mse": independent_value,
            "shared_fit_mse": final_value,
            "cluster_fit_mse": final_value,
            "zero_fit_mse": zero_value,
            "shared_relative_gain_vs_zero": float(relative_gain),
            "shared_fit_mse_regression_vs_independent": float(regression),
            "cluster_fit_mse_regression_vs_independent": float(regression),
            "final_mag_loss": float(final_mag.detach().cpu().item()),
            "final_sh_mag_loss": float(final_sh_mag.detach().cpu().item()),
            "final_scale_reg": float(final_scale_reg.detach().cpu().item()),
            "passed": passed,
            "applied": passed,
        }
    )
    if mode == "rank2" and components is not None and face_mix is not None:
        component_flat = components.detach().reshape(int(components.shape[0]), -1)
        mix_sums = face_mix.sum(dim=-1)
        face_mix_cpu = face_mix.detach().cpu().tolist()
        result.update(
            {
                "component_count": int(components.shape[0]),
                "component_coefficients": components.detach().cpu().tolist(),
                "face_mixture_min": float(face_mix.min().detach().cpu().item()),
                "face_mixture_max": float(face_mix.max().detach().cpu().item()),
                "face_mixture_sum_min": float(mix_sums.min().detach().cpu().item()),
                "face_mixture_sum_max": float(mix_sums.max().detach().cpu().item()),
                "face_mixture_simplex_max_abs_error": float((mix_sums - 1.0).abs().max().detach().cpu().item()),
                "face_mixture_entropy": float(
                    (-(face_mix * face_mix.clamp_min(1e-8).log()).sum(dim=1).mean()).detach().cpu().item()
                ),
                "component_l2": [float(v) for v in component_flat.norm(dim=1).detach().cpu().tolist()],
                "component_max_abs": [float(v) for v in component_flat.abs().max(dim=1).values.detach().cpu().tolist()],
                "face_mixtures": [
                    {"face_id": int(fid), "weights": [float(x) for x in face_mix_cpu[idx]]}
                    for idx, fid in enumerate(face_ids)
                ],
                "final_mixture_reg": float(final_scale_reg.detach().cpu().item()),
            }
        )
    if mode in {"chart_linear", "chart_quad", "field_linear", "field_quad"} and chart_params is not None and chart_features is not None:
        chart_flat = chart_params.detach().reshape(int(chart_params.shape[0]), -1)
        result.update(
            {
                "chart_feature_count": int(chart_params.shape[0]),
                "chart_scale": float(chart_uv_scale),
                "chart_uv_scale": float(chart_uv_scale),
                "chart_svd_scale": float(chart_svd_scale),
                "chart_feature_min": float(chart_features[..., 1:].min().detach().cpu().item()),
                "chart_feature_max": float(chart_features[..., 1:].max().detach().cpu().item()),
                "chart_component_l2": [float(v) for v in chart_flat.norm(dim=1).detach().cpu().tolist()],
                "chart_component_max_abs": [float(v) for v in chart_flat.abs().max(dim=1).values.detach().cpu().tolist()],
                "chart_coefficients": chart_params.detach().cpu().tolist(),
                "face_chart_features": [
                    {"face_id": int(fid), "corner_features": chart_features[idx].detach().cpu().tolist()}
                    for idx, fid in enumerate(face_ids)
                ],
                "final_chart_linear_reg": float(final_scale_reg.detach().cpu().item()),
            }
        )
    if face_scale is not None:
        face_scale_cpu = face_scale.detach().cpu().tolist()
        result.update(
            {
                "face_scale_min": float(face_scale.min().detach().cpu().item()),
                "face_scale_mean": float(face_scale.mean().detach().cpu().item()),
                "face_scale_max": float(face_scale.max().detach().cpu().item()),
                "face_scales": [
                    {"face_id": int(fid), "scale": float(face_scale_cpu[idx])}
                    for idx, fid in enumerate(face_ids)
                ],
            }
        )
    if not passed:
        result["rejected_reason"] = "shared_basis_fit_regression"
        return result

    clamped_count = 0
    total_coeff_count = 0
    max_clamp_excess = 0.0
    for fid, row in zip(face_ids, local_rows):
        if mode in {"chart_linear", "chart_quad", "field_linear", "field_quad"} and chart_params is not None and chart_features is not None:
            materialized = torch.einsum(
                "cf,fbr->cbr",
                chart_features[face_to_patch_row[int(fid)]],
                chart_params,
            )
        elif mode == "rank2" and components is not None and face_mix is not None:
            materialized = (components * face_mix[face_to_patch_row[int(fid)]].view(-1, 1, 1, 1)).sum(dim=0)
        else:
            assert shared is not None
            materialized = shared
        if face_scale is not None:
            materialized = materialized * face_scale[face_to_patch_row[int(fid)]].view(1, 1, 1)
        excess = materialized.abs() - bounds
        clamped_count += int((excess > 1e-7).sum().detach().cpu().item())
        total_coeff_count += int(materialized.numel())
        if excess.numel():
            max_clamp_excess = max(max_clamp_excess, float(excess.clamp_min(0).max().detach().cpu().item()))
        materialized = torch.clamp(materialized, min=-bounds, max=bounds)
        coeff[row * 3 : row * 3 + 3, :basis_count, :] = materialized.to(dtype=coeff.dtype)
    result.update(
        {
            "effective_max_scale": max_scale if face_scale is not None else 1.0,
            "coeff_clamped_count": int(clamped_count),
            "coeff_total_count": int(total_coeff_count),
            "coeff_clamped_fraction": float(clamped_count / max(total_coeff_count, 1)),
            "coeff_max_clamp_excess": float(max_clamp_excess),
        }
    )
    return result


def grow_patch_certified_faces(
    *,
    coeff: torch.Tensor,
    faces: torch.Tensor,
    vertices: torch.Tensor,
    selected_faces: list[int],
    seed_faces: list[int],
    face_stats: dict[int, dict[str, float]],
    face_policy: dict[int, dict[str, float]],
    fit_ids: torch.Tensor,
    fit_basis: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    fit_samples: PixelSamples,
    val_ids: torch.Tensor,
    val_basis: torch.Tensor,
    val_target: torch.Tensor,
    val_weights: torch.Tensor,
    val_samples: PixelSamples,
    patch_crossfold_cache: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any], dict[int, dict[str, Any]]]:
    rings = int(args.patch_cert_rings)
    patch_crossfold_enabled = bool(int(args.patch_cert_crossfold_folds) > 1)
    patch_neighbor_crossfold = bool(args.patch_cert_neighbor_crossfold) and patch_crossfold_enabled
    summary: dict[str, Any] = {
        "enabled": bool(rings > 0),
        "rings": max(rings, 0),
        "max_faces_per_seed": int(args.patch_cert_max_faces_per_seed),
        "min_direction_cosine": float(args.patch_cert_min_direction_cosine),
        "min_neighbor_policy_val_samples": int(args.patch_cert_min_neighbor_policy_val_samples),
        "min_neighbor_policy_val_relative_gain": float(args.patch_cert_min_neighbor_policy_val_relative_gain),
        "min_policy_val_samples": int(args.patch_cert_min_policy_val_samples),
        "min_relative_gain": float(args.patch_cert_min_relative_gain),
        "neighbor_mode": str(args.patch_cert_neighbor_mode),
        "centroid_candidates_per_seed": int(args.patch_cert_centroid_candidates_per_seed),
        "patch_shrink": bool(args.patch_cert_shrink),
        "patch_cluster_basis": bool(args.patch_cert_cluster_basis),
        "patch_cluster_basis_mode": str(args.patch_cert_cluster_basis_mode),
        "patch_cluster_basis_steps": int(args.patch_cert_cluster_basis_steps),
        "patch_cluster_basis_min_samples": int(args.patch_cert_cluster_basis_min_samples),
        "patch_cluster_basis_max_scale": float(args.patch_cert_cluster_basis_max_scale),
        "patch_cluster_basis_max_fit_mse_regression": float(args.patch_cert_cluster_basis_max_fit_mse_regression),
        "patch_crossfold_enabled": patch_crossfold_enabled,
        "patch_crossfold_folds": int(args.patch_cert_crossfold_folds),
        "patch_crossfold_min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "patch_crossfold_min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "patch_crossfold_min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "patch_neighbor_crossfold": patch_neighbor_crossfold,
        "seed_faces": int(len(seed_faces)),
        "accepted_faces_before": int(len(seed_faces)),
        "accepted_faces_after": int(len(seed_faces)),
        "accepted_patches": 0,
        "rejected_patches": 0,
        "rejected_patch_crossfold": 0,
        "rejected_neighbor_crossfold": 0,
        "rejected_cluster_basis": 0,
        "rejected_patch_budget": 0,
        "rejected_post_shrink_policy_val": 0,
        "accepted_patch_crossfold": 0,
        "accepted_cluster_basis": 0,
        "accepted_post_shrink_policy_val": 0,
        "accepted_post_shrink_patch_crossfold": 0,
        "mean_patch_size": 1.0 if seed_faces else 0.0,
        "preview": [],
    }
    if rings <= 0 or not seed_faces:
        return list(seed_faces), summary, {}

    adjacency = selected_face_adjacency(faces, selected_faces)
    centroid_face_ids, centroid_centers, centroid_index = selected_face_centers(faces, vertices, selected_faces)
    selected_set = set(int(fid) for fid in selected_faces)
    assigned: set[int] = set()
    accepted: list[int] = []
    patch_by_face: dict[int, dict[str, Any]] = {}
    patch_sizes: list[int] = []

    for seed in seed_faces:
        seed_id = int(seed)
        if seed_id in assigned:
            continue
        patch: list[int] = [seed_id]
        seen = {seed_id}
        frontier = [seed_id]
        for _ in range(rings):
            next_frontier: list[int] = []
            for fid in frontier:
                neighbors: list[int] = []
                if str(args.patch_cert_neighbor_mode) in {"topology", "both"}:
                    neighbors.extend(sorted(adjacency.get(int(fid), set())))
                if int(fid) == seed_id and str(args.patch_cert_neighbor_mode) in {"centroid", "both"}:
                    neighbors.extend(
                        centroid_neighbor_candidates(
                            seed_id,
                            centroid_face_ids,
                            centroid_centers,
                            centroid_index,
                            int(args.patch_cert_centroid_candidates_per_seed),
                        )
                    )
                deduped_neighbors = []
                seen_neighbors: set[int] = set()
                for nb in neighbors:
                    if int(nb) in seen_neighbors:
                        continue
                    seen_neighbors.add(int(nb))
                    deduped_neighbors.append(int(nb))
                for nb in deduped_neighbors:
                    nb = int(nb)
                    if nb in seen or nb in assigned or nb not in selected_set:
                        continue
                    stats = face_stats.get(nb, {})
                    if int(stats.get("view_hits", 0)) < int(args.min_view_hits):
                        continue
                    if float(stats.get("pixel_count", 0.0)) < float(args.min_pixel_count):
                        continue
                    if residual_direction_cosine(face_stats, seed_id, nb) < float(args.patch_cert_min_direction_cosine):
                        continue
                    proxy = face_policy.get(nb, {})
                    if int(proxy.get("samples", 0)) < int(args.patch_cert_min_neighbor_policy_val_samples):
                        continue
                    if float(proxy.get("relative_gain", -1.0)) < float(args.patch_cert_min_neighbor_policy_val_relative_gain):
                        continue
                    if patch_neighbor_crossfold:
                        neighbor_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, [nb], args)
                        if not bool(neighbor_crossfold.get("passed", False)):
                            summary["rejected_neighbor_crossfold"] = int(summary["rejected_neighbor_crossfold"]) + 1
                            continue
                    patch.append(nb)
                    seen.add(nb)
                    next_frontier.append(nb)
                    if len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                        break
                if len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                    break
            frontier = next_frontier
            if not frontier or len(patch) >= max(int(args.patch_cert_max_faces_per_seed), 1):
                break

        coeff_snapshot = clone_face_coeffs(coeff, selected_faces, patch)
        patch_cluster_basis = fit_patch_cluster_shared_basis(
            coeff,
            selected_faces,
            patch,
            fit_ids,
            fit_basis,
            fit_target,
            fit_weights,
            fit_samples.face_ids,
            fit_samples.view_names,
            args,
            faces=faces,
            vertices=vertices,
        )
        proxy_before_shrink = evaluate_proxy_for_faces(
            coeff,
            val_ids,
            val_basis,
            val_target,
            val_weights,
            val_samples.face_ids,
            patch,
        )
        passed = (
            int(proxy_before_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
            and float(proxy_before_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
        )
        patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
        cluster_basis_failed = bool(args.patch_cert_cluster_basis) and not bool(patch_cluster_basis.get("passed", False))
        if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
            passed = False
        if cluster_basis_failed:
            passed = False
        if not passed:
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
                summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
            if cluster_basis_failed:
                summary["rejected_cluster_basis"] = int(summary["rejected_cluster_basis"]) + 1
            patch = [seed_id]
            coeff_snapshot = clone_face_coeffs(coeff, selected_faces, patch)
            patch_cluster_basis = fit_patch_cluster_shared_basis(
                coeff,
                selected_faces,
                patch,
                fit_ids,
                fit_basis,
                fit_target,
                fit_weights,
                fit_samples.face_ids,
                fit_samples.view_names,
                args,
                faces=faces,
                vertices=vertices,
            )
            proxy_before_shrink = evaluate_proxy_for_faces(
                coeff,
                val_ids,
                val_basis,
                val_target,
                val_weights,
                val_samples.face_ids,
                patch,
            )
            patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
            if patch_crossfold_enabled:
                passed = (
                    int(proxy_before_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
                    and float(proxy_before_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
                    and bool(patch_crossfold.get("passed", False))
                )
            else:
                # Preserve historical PatchCert behavior when the new patch-fold
                # certificate is disabled: a rejected grown patch falls back to
                # its already face-certified seed instead of rejecting the seed.
                passed = False
            if patch_crossfold_enabled and not passed:
                if patch_crossfold_enabled and not bool(patch_crossfold.get("passed", False)):
                    summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
                patch_record = {
                    "seed_face": seed_id,
                    "faces": [seed_id],
                    "patch_size": 1,
                    "proxy": proxy_before_shrink,
                    "proxy_before_shrink": proxy_before_shrink,
                    "passed_patch_gain": False,
                    "patch_crossfold_certificate": patch_crossfold,
                    "patch_cluster_basis": patch_cluster_basis,
                    "scale": 0.0,
                    "raw_scale": 0.0,
                    "scale_samples": int(proxy_before_shrink.get("samples", 0)),
                    "rejected": True,
                }
                if len(summary["preview"]) < 20:
                    summary["preview"].append(patch_record)
                continue

        scale = 1.0
        raw_scale = 1.0
        scale_samples = int(proxy_before_shrink.get("samples", 0))
        if bool(args.patch_cert_shrink) and len(patch) > 1:
            scale, scale_samples, raw_scale = fit_patch_scale(
                coeff,
                val_ids,
                val_basis,
                val_target,
                val_weights,
                val_samples.face_ids,
                patch,
                int(args.patch_cert_min_policy_val_samples),
            )
            scale_face_coeffs(coeff, selected_faces, patch, scale)
        post_shrink_patch_crossfold = patch_crossfold_certificate_for_faces(coeff, patch_crossfold_cache, patch, args)
        if patch_crossfold_enabled and not bool(post_shrink_patch_crossfold.get("passed", False)):
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_patch_crossfold"] = int(summary["rejected_patch_crossfold"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_before_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "patch_cluster_basis": patch_cluster_basis,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "post_shrink_patch_crossfold_failed",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue
        proxy_after_shrink = evaluate_proxy_for_faces(
            coeff,
            val_ids,
            val_basis,
            val_target,
            val_weights,
            val_samples.face_ids,
            patch,
        )
        post_shrink_policy_pass = (
            int(proxy_after_shrink.get("samples", 0)) >= int(args.patch_cert_min_policy_val_samples)
            and float(proxy_after_shrink.get("relative_gain", -1.0)) >= float(args.patch_cert_min_relative_gain)
        )
        if not post_shrink_policy_pass:
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_post_shrink_policy_val"] = int(summary["rejected_post_shrink_policy_val"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_after_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "patch_cluster_basis": patch_cluster_basis,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "post_shrink_policy_val_failed",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue
        budget = int(args.max_faces_to_apply)
        if budget >= 0 and len(accepted) + len(patch) > budget:
            restore_face_coeffs(coeff, selected_faces, coeff_snapshot)
            summary["rejected_patches"] = int(summary["rejected_patches"]) + 1
            summary["rejected_patch_budget"] = int(summary["rejected_patch_budget"]) + 1
            patch_record = {
                "seed_face": seed_id,
                "faces": [int(fid) for fid in patch],
                "patch_size": int(len(patch)),
                "proxy": proxy_after_shrink,
                "proxy_before_shrink": proxy_before_shrink,
                "passed_patch_gain": False,
                "patch_crossfold_certificate": patch_crossfold,
                "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
                "patch_cluster_basis": patch_cluster_basis,
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "scale_samples": int(scale_samples),
                "rejected": True,
                "rejected_reason": "patch_budget_would_split_carrier",
            }
            if len(summary["preview"]) < 20:
                summary["preview"].append(patch_record)
            continue

        assigned.update(int(fid) for fid in patch)
        accepted.extend(int(fid) for fid in patch)
        patch_sizes.append(len(patch))
        if len(patch) > 1:
            summary["accepted_patches"] = int(summary["accepted_patches"]) + 1
        if bool(args.patch_cert_cluster_basis) and bool(patch_cluster_basis.get("applied", False)):
            summary["accepted_cluster_basis"] = int(summary["accepted_cluster_basis"]) + 1
        if patch_crossfold_enabled and bool(patch_crossfold.get("passed", False)):
            summary["accepted_patch_crossfold"] = int(summary["accepted_patch_crossfold"]) + 1
        if post_shrink_policy_pass:
            summary["accepted_post_shrink_policy_val"] = int(summary["accepted_post_shrink_policy_val"]) + 1
        if patch_crossfold_enabled and bool(post_shrink_patch_crossfold.get("passed", False)):
            summary["accepted_post_shrink_patch_crossfold"] = int(summary["accepted_post_shrink_patch_crossfold"]) + 1
        patch_record = {
            "seed_face": seed_id,
            "faces": [int(fid) for fid in patch],
            "patch_size": int(len(patch)),
            "proxy": proxy_after_shrink,
            "proxy_before_shrink": proxy_before_shrink,
            "passed_patch_gain": bool(passed),
            "patch_crossfold_certificate": patch_crossfold,
            "post_shrink_patch_crossfold_certificate": post_shrink_patch_crossfold,
            "patch_cluster_basis": patch_cluster_basis,
            "scale": float(scale),
            "raw_scale": float(raw_scale),
            "scale_samples": int(scale_samples),
        }
        for fid in patch:
            patch_by_face[int(fid)] = patch_record
        if len(summary["preview"]) < 20:
            summary["preview"].append(patch_record)

    if patch_sizes:
        summary["mean_patch_size"] = float(np.mean(np.asarray(patch_sizes, dtype=np.float32)))
    summary["accepted_faces_after"] = int(len(accepted))
    return accepted, summary, patch_by_face


def solve_coeff_delta(
    selected_faces_local: torch.Tensor,
    fit_sample_vertex_ids: torch.Tensor,
    fit_weighted_basis: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    *,
    vertex_count: int,
    max_abs_dc_coeff: float,
    max_abs_sh_coeff: float,
    lambda_mag: float,
    lambda_sh1_mag: float,
    lambda_smooth: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    basis_count = int(fit_weighted_basis.shape[2]) if fit_weighted_basis.ndim == 3 else 4
    if vertex_count <= 0 or fit_sample_vertex_ids.numel() == 0:
        return torch.empty((0, basis_count, 3), dtype=torch.float32), {
            "initial_fit_mse": 0.0,
            "final_fit_mse": 0.0,
            "final_mag_loss": 0.0,
            "final_sh_mag_loss": 0.0,
            "final_smooth_loss": 0.0,
            "basis_count": int(basis_count),
        }
    fit_sample_vertex_ids = fit_sample_vertex_ids.to(device=device)
    fit_weighted_basis = fit_weighted_basis.to(device=device)
    fit_target = fit_target.to(device=device)
    fit_weights = fit_weights.to(device=device).clamp_min(1e-8)
    selected_faces_local = selected_faces_local.to(device=device)
    edges = surface_edges(selected_faces_local.detach().cpu()).to(device=device)
    bounds = torch.full((basis_count,), float(max_abs_sh_coeff), dtype=torch.float32, device=device)
    bounds[0] = float(max_abs_dc_coeff)
    bounds = bounds.view(1, basis_count, 1)
    param = torch.zeros((int(vertex_count), basis_count, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    with torch.no_grad():
        zero = torch.zeros_like(param)
        initial_fit_mse = _weighted_mse(zero, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)

    final_fit_mse = initial_fit_mse
    final_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_sh_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
    for _ in range(int(steps)):
        coeff = bounds * torch.tanh(param)
        data_loss = _weighted_mse(coeff, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)
        mag_loss = (coeff[:, 0, :] ** 2).mean()
        sh_mag_loss = (coeff[:, 1:, :] ** 2).mean() if basis_count > 1 else torch.zeros((), dtype=torch.float32, device=device)
        if edges.numel():
            smooth_loss = ((coeff[edges[:, 0]] - coeff[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = (
            data_loss
            + float(lambda_mag) * mag_loss
            + float(lambda_sh1_mag) * sh_mag_loss
            + float(lambda_smooth) * smooth_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_fit_mse = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_sh_mag_loss = sh_mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()

    with torch.no_grad():
        coeff = (bounds * torch.tanh(param)).detach().cpu()
    return coeff, {
        "initial_fit_mse": float(initial_fit_mse.detach().cpu().item()),
        "final_fit_mse": float(final_fit_mse.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_sh_mag_loss": float(final_sh_mag_loss.detach().cpu().item()),
        "basis_count": int(basis_count),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
    }


def build_shared_residual_field_features(
    vertices_local: torch.Tensor,
    *,
    anchor_count: int,
    sigma: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build deterministic low-dimensional RBF features for local mesh slots."""

    if vertices_local.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float32), {
            "enabled": True,
            "anchor_count_requested": int(anchor_count),
            "anchor_count": 0,
            "feature_count": 0,
            "sigma": 0.0,
        }
    points = vertices_local.detach().cpu().float()
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    scale = centered.norm(dim=1).max().clamp_min(1.0e-6)
    norm_points = centered / scale
    n = int(norm_points.shape[0])
    k = min(max(int(anchor_count), 1), n)
    anchor_indices: list[int] = [0]
    if k > 1:
        dist2 = ((norm_points - norm_points[0:1]) ** 2).sum(dim=1)
        for _ in range(1, k):
            idx = int(torch.argmax(dist2).item())
            anchor_indices.append(idx)
            next_dist2 = ((norm_points - norm_points[idx : idx + 1]) ** 2).sum(dim=1)
            dist2 = torch.minimum(dist2, next_dist2)
    anchors = norm_points[torch.as_tensor(anchor_indices, dtype=torch.long)]
    if float(sigma) > 0.0:
        sigma_value = float(sigma)
    elif k > 1:
        pairwise = torch.cdist(anchors, anchors).reshape(-1)
        pairwise = pairwise[pairwise > 1.0e-6]
        sigma_value = float(pairwise.median().item()) if pairwise.numel() else 1.0
        sigma_value = max(sigma_value, 0.15)
    else:
        sigma_value = 1.0
    rbf = torch.exp(-torch.cdist(norm_points, anchors) ** 2 / (2.0 * max(sigma_value, 1.0e-6) ** 2))
    rbf = rbf / rbf.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    features = torch.cat([torch.ones((n, 1), dtype=torch.float32), norm_points, rbf], dim=1)
    meta = {
        "enabled": True,
        "basis_type": "global_rbf_shared_residual_field",
        "anchor_count_requested": int(anchor_count),
        "anchor_count": int(k),
        "feature_count": int(features.shape[1]),
        "sigma": float(sigma_value),
        "normalization_center": [float(v) for v in center.reshape(-1).tolist()],
        "normalization_scale": float(scale.item()),
        "anchor_local_indices": [int(i) for i in anchor_indices],
        "anchor_points_normalized": anchors.tolist(),
    }
    return features, meta


def _view_hinge_loss(
    coeff: torch.Tensor,
    sample_vertex_ids: torch.Tensor,
    weighted_basis: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    view_names: list[str],
    *,
    min_samples: int,
) -> tuple[torch.Tensor, int]:
    if sample_vertex_ids.numel() == 0 or not view_names:
        return torch.zeros((), dtype=torch.float32, device=coeff.device), 0
    view_np = np.asarray(view_names, dtype=object)
    losses: list[torch.Tensor] = []
    zero = torch.zeros_like(coeff)
    for view_name in sorted(set(str(v) for v in view_np.tolist())):
        mask_np = view_np == view_name
        if int(mask_np.sum()) < max(int(min_samples), 1):
            continue
        idx = torch.as_tensor(np.nonzero(mask_np)[0], dtype=torch.long, device=sample_vertex_ids.device)
        mse_before = _weighted_mse(
            zero,
            sample_vertex_ids[idx],
            weighted_basis[idx],
            target[idx],
            weights[idx],
        ).detach()
        mse_after = _weighted_mse(
            coeff,
            sample_vertex_ids[idx],
            weighted_basis[idx],
            target[idx],
            weights[idx],
        )
        losses.append(torch.relu((mse_after - mse_before) / mse_before.clamp_min(1.0e-12)))
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=coeff.device), 0
    return torch.stack(losses).mean(), len(losses)


def _duplicate_source_smooth_loss(coeff: torch.Tensor, source_vertex_ids: torch.Tensor) -> torch.Tensor:
    if coeff.numel() == 0 or source_vertex_ids.numel() != int(coeff.shape[0]):
        return torch.zeros((), dtype=torch.float32, device=coeff.device)
    source = source_vertex_ids.to(device=coeff.device, dtype=torch.long)
    _, inverse, counts = torch.unique(source, return_inverse=True, return_counts=True)
    repeated = counts[inverse] > 1
    if not bool(repeated.any().item()):
        return torch.zeros((), dtype=torch.float32, device=coeff.device)
    means = torch.zeros((int(counts.shape[0]), int(coeff.shape[1]), int(coeff.shape[2])), dtype=coeff.dtype, device=coeff.device)
    means.index_add_(0, inverse, coeff)
    means = means / counts.to(device=coeff.device, dtype=coeff.dtype).view(-1, 1, 1).clamp_min(1.0)
    return ((coeff[repeated] - means[inverse[repeated]]) ** 2).mean()


def solve_shared_residual_field_delta(
    selected_faces_local: torch.Tensor,
    source_vertex_ids: torch.Tensor,
    vertices_local: torch.Tensor,
    fit_sample_vertex_ids: torch.Tensor,
    fit_weighted_basis: torch.Tensor,
    fit_target: torch.Tensor,
    fit_weights: torch.Tensor,
    fit_view_names: list[str],
    *,
    vertex_count: int,
    max_abs_dc_coeff: float,
    max_abs_sh_coeff: float,
    lambda_mag: float,
    lambda_sh1_mag: float,
    lambda_smooth: float,
    steps: int,
    lr: float,
    anchor_count: int,
    sigma: float,
    weight_l2: float,
    view_hinge_weight: float,
    view_hinge_min_samples: int,
    duplicate_smooth_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    basis_count = int(fit_weighted_basis.shape[2]) if fit_weighted_basis.ndim == 3 else 4
    if vertex_count <= 0 or fit_sample_vertex_ids.numel() == 0:
        return torch.empty((0, basis_count, 3), dtype=torch.float32), {
            "solver_type": "shared_residual_field",
            "initial_fit_mse": 0.0,
            "final_fit_mse": 0.0,
            "basis_count": int(basis_count),
            "shared_residual_field": {"enabled": True, "blocked_reason": "empty_inputs"},
        }

    features_cpu, field_meta = build_shared_residual_field_features(
        vertices_local,
        anchor_count=int(anchor_count),
        sigma=float(sigma),
    )
    fit_sample_vertex_ids = fit_sample_vertex_ids.to(device=device)
    fit_weighted_basis = fit_weighted_basis.to(device=device)
    fit_target = fit_target.to(device=device)
    fit_weights = fit_weights.to(device=device).clamp_min(1e-8)
    selected_faces_local = selected_faces_local.to(device=device)
    source_vertex_ids = source_vertex_ids.to(device=device)
    features = features_cpu.to(device=device)
    edges = surface_edges(selected_faces_local.detach().cpu()).to(device=device)
    bounds = torch.full((basis_count,), float(max_abs_sh_coeff), dtype=torch.float32, device=device)
    bounds[0] = float(max_abs_dc_coeff)
    bounds = bounds.view(1, basis_count, 1)

    param = torch.zeros((int(features.shape[1]), basis_count, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=float(lr))

    def coeff_from_param() -> torch.Tensor:
        raw = torch.einsum("vf,fbc->vbc", features, param)
        return bounds * torch.tanh(raw)

    with torch.no_grad():
        zero = torch.zeros((int(vertex_count), basis_count, 3), dtype=torch.float32, device=device)
        initial_fit_mse = _weighted_mse(zero, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)

    final_fit_mse = initial_fit_mse
    final_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_sh_mag_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
    final_weight_l2 = torch.zeros((), dtype=torch.float32, device=device)
    final_view_hinge = torch.zeros((), dtype=torch.float32, device=device)
    final_duplicate_smooth = torch.zeros((), dtype=torch.float32, device=device)
    view_hinge_groups = 0
    for _ in range(int(steps)):
        coeff = coeff_from_param()
        data_loss = _weighted_mse(coeff, fit_sample_vertex_ids, fit_weighted_basis, fit_target, fit_weights)
        mag_loss = (coeff[:, 0, :] ** 2).mean()
        sh_mag_loss = (coeff[:, 1:, :] ** 2).mean() if basis_count > 1 else torch.zeros((), dtype=torch.float32, device=device)
        if edges.numel():
            smooth_loss = ((coeff[edges[:, 0]] - coeff[edges[:, 1]]) ** 2).mean()
        else:
            smooth_loss = torch.zeros((), dtype=torch.float32, device=device)
        duplicate_smooth = _duplicate_source_smooth_loss(coeff, source_vertex_ids)
        view_hinge, view_hinge_groups = _view_hinge_loss(
            coeff,
            fit_sample_vertex_ids,
            fit_weighted_basis,
            fit_target,
            fit_weights,
            fit_view_names,
            min_samples=int(view_hinge_min_samples),
        )
        weight_l2_loss = (param**2).mean()
        loss = (
            data_loss
            + float(lambda_mag) * mag_loss
            + float(lambda_sh1_mag) * sh_mag_loss
            + float(lambda_smooth) * smooth_loss
            + float(duplicate_smooth_weight) * duplicate_smooth
            + float(view_hinge_weight) * view_hinge
            + float(weight_l2) * weight_l2_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_fit_mse = data_loss.detach()
        final_mag_loss = mag_loss.detach()
        final_sh_mag_loss = sh_mag_loss.detach()
        final_smooth_loss = smooth_loss.detach()
        final_weight_l2 = weight_l2_loss.detach()
        final_view_hinge = view_hinge.detach()
        final_duplicate_smooth = duplicate_smooth.detach()

    with torch.no_grad():
        coeff = coeff_from_param().detach().cpu()
    field_meta.update(
        {
            "param_count": int(param.numel()),
            "weight_l2": float(weight_l2),
            "view_hinge_weight": float(view_hinge_weight),
            "view_hinge_min_samples": int(view_hinge_min_samples),
            "view_hinge_groups": int(view_hinge_groups),
            "duplicate_smooth_weight": float(duplicate_smooth_weight),
        }
    )
    return coeff, {
        "solver_type": "shared_residual_field",
        "initial_fit_mse": float(initial_fit_mse.detach().cpu().item()),
        "final_fit_mse": float(final_fit_mse.detach().cpu().item()),
        "final_mag_loss": float(final_mag_loss.detach().cpu().item()),
        "final_sh_mag_loss": float(final_sh_mag_loss.detach().cpu().item()),
        "basis_count": int(basis_count),
        "final_smooth_loss": float(final_smooth_loss.detach().cpu().item()),
        "final_weight_l2_loss": float(final_weight_l2.detach().cpu().item()),
        "final_view_hinge_loss": float(final_view_hinge.detach().cpu().item()),
        "final_duplicate_source_smooth_loss": float(final_duplicate_smooth.detach().cpu().item()),
        "shared_residual_field": field_meta,
    }


def samples_to_tensors(
    samples: PixelSamples,
    sample_vertex_ids: torch.Tensor,
    vertices_local: torch.Tensor,
    *,
    strength: float,
    max_abs_delta_rgb: float,
    sh_degree: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sample_vertex_ids.to(device=device)
    bary = torch.as_tensor(samples.barycentric, dtype=torch.float32, device=device)
    centers = torch.as_tensor(samples.camera_centers, dtype=torch.float32, device=device)
    vertices_local = vertices_local.to(device=device)
    weighted_basis = _sh_basis(vertices_local, ids, bary, centers, degree=int(sh_degree))
    target = torch.as_tensor(samples.residual_rgb, dtype=torch.float32, device=device)
    target = (target * float(strength)).clamp(-float(max_abs_delta_rgb), float(max_abs_delta_rgb))
    weights = torch.as_tensor(samples.weights, dtype=torch.float32, device=device)
    return ids, weighted_basis, target, weights


def materialize_facelocal(
    state: dict[str, Any],
    faces: torch.Tensor,
    selected_faces: list[int],
    source_vertex_ids: torch.Tensor,
    coeff: torch.Tensor,
    accepted_faces: list[int],
) -> dict[str, Any]:
    out = clone_state(state)
    if not accepted_faces:
        return out
    vertex_count = int(state["triangles_points"].shape[0])
    face_count = int(faces.shape[0])
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    accepted_local_rows = [face_to_selected[int(fid)] for fid in accepted_faces]
    local_vertex_indices: list[int] = []
    for row in accepted_local_rows:
        local_vertex_indices.extend([row * 3, row * 3 + 1, row * 3 + 2])
    local_idx = torch.as_tensor(local_vertex_indices, dtype=torch.long)
    source_idx = source_vertex_ids[local_idx].long()
    coeff_add = coeff[local_idx]

    new_faces = faces.clone()
    start = vertex_count
    for out_row, fid in enumerate(accepted_faces):
        new_faces[int(fid)] = torch.tensor([start + out_row * 3, start + out_row * 3 + 1, start + out_row * 3 + 2])

    for key, value in state.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        cpu = value.detach().cpu()
        if key == "_triangle_indices":
            out[key] = new_faces.to(dtype=value.dtype)
        elif cpu.ndim > 0 and int(cpu.shape[0]) == vertex_count:
            append = cpu[source_idx].clone()
            if key == "features_dc":
                append = append + coeff_add[:, 0:1, :].to(dtype=append.dtype)
            elif key == "features_rest":
                append = append.clone()
                if append.ndim == 3 and append.shape[1] > 0 and coeff_add.shape[1] > 1:
                    rest_count = min(int(append.shape[1]), int(coeff_add.shape[1]) - 1)
                    append[:, :rest_count, :] = append[:, :rest_count, :] + coeff_add[:, 1 : 1 + rest_count, :].to(
                        dtype=append.dtype
                    )
            out[key] = torch.cat([cpu, append], dim=0).to(dtype=value.dtype)
        elif cpu.ndim > 0 and int(cpu.shape[0]) == face_count:
            out[key] = cpu.clone().to(dtype=value.dtype)
        else:
            out[key] = cpu.clone()
    return out


def _parse_face_id_filter(raw: str) -> set[int]:
    out: set[int] = set()
    for item in str(raw or "").replace(" ", ",").split(","):
        if not item:
            continue
        out.add(int(item))
    return out


def read_candidate_plan(
    path: Path,
    *,
    limit: int = 0,
    face_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        meta = payload
        if isinstance(payload.get("candidates"), list):
            rows = payload["candidates"]
        elif isinstance(payload.get("accepted"), list):
            rows = payload["accepted"]
        elif isinstance(payload.get("accepted_preview"), list):
            rows = payload["accepted_preview"]
        else:
            rows = []
    else:
        rows = []
    filtered: list[dict[str, Any]] = []
    keep_ids = face_ids or set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            continue
        if keep_ids and face_id not in keep_ids:
            continue
        filtered.append(dict(row))
    if int(limit) > 0:
        filtered = filtered[: int(limit)]
    return filtered, meta


def read_plan_alphas(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("face_alphas", payload.get("alphas", payload)) if isinstance(payload, dict) else payload
    alphas: dict[int, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                face_id = int(key)
                alpha = float(value)
            except Exception:
                continue
            if math.isfinite(alpha):
                alphas[face_id] = alpha
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                face_id = int(item.get("face_id"))
                alpha = float(item.get("alpha", item.get("scale", 1.0)))
            except Exception:
                continue
            if math.isfinite(alpha):
                alphas[face_id] = alpha
    return alphas


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_strict_materialize_request(args: argparse.Namespace) -> None:
    errors: list[str] = []
    scale = float(args.materialize_plan_scale)
    if int(args.materialize_plan_limit) > 0:
        errors.append("materialize_plan_limit would row-slice a certified carrier")
    if str(args.materialize_plan_face_ids or "").strip():
        errors.append("materialize_plan_face_ids would subset a certified carrier")
    render_trust_path = args.materialize_plan_render_trust_json
    if not math.isfinite(scale):
        errors.append("materialize_plan_scale would alter certified coefficients")
    elif abs(scale - 1.0) > 1e-12 and render_trust_path is None:
        errors.append("materialize_plan_scale would alter certified coefficients without a render-trust certificate")
    if args.materialize_plan_alpha_json is not None:
        errors.append("materialize_plan_alpha_json would alter certified coefficients")
    if errors:
        raise ValueError(
            "Strict certified plan materialization rejected unsafe replay controls: "
            + "; ".join(errors)
            + ". Use --materialize_plan_render_trust_json for audited train-val render-certified scale replay, "
            + "or --materialize_allow_uncertified_plan only for explicitly labeled legacy ablations."
        )


def validate_render_trust_certificate(
    *,
    cert_path: Path | None,
    plan_path: Path,
    requested_scale: float,
) -> dict[str, Any]:
    if cert_path is None:
        if abs(float(requested_scale) - 1.0) <= 1e-12:
            return {"enabled": False}
        raise ValueError("non-unit strict materialize_plan_scale requires --materialize_plan_render_trust_json")
    payload = json.loads(cert_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--materialize_plan_render_trust_json must contain a JSON object")
    errors: list[str] = []
    if not bool(payload.get("accepted", False)):
        errors.append("render_trust_certificate_not_accepted")
    if bool(payload.get("selection_uses_test", True)):
        errors.append("render_trust_certificate_used_test")
    try:
        accepted_scale = float(payload.get("accepted_scale", payload.get("scale")))
    except Exception:
        accepted_scale = float("nan")
    if not math.isfinite(accepted_scale) or abs(accepted_scale - float(requested_scale)) > 1e-9:
        errors.append("render_trust_scale_mismatch")
    expected_sha = str(payload.get("plan_sha256", "")).strip()
    actual_sha = file_sha256(plan_path)
    if expected_sha and expected_sha != actual_sha:
        errors.append("render_trust_plan_sha256_mismatch")
    if errors:
        raise ValueError(
            "Strict certified plan materialization rejected render-trust certificate: "
            + "; ".join(errors)
        )
    return {
        "enabled": True,
        "certificate": str(cert_path),
        "accepted_scale": float(accepted_scale),
        "plan_sha256": actual_sha,
        "selection_uses_test": bool(payload.get("selection_uses_test", True)),
        "accepted": bool(payload.get("accepted", False)),
        "trainval_balanced_delta": payload.get("trainval_balanced_delta"),
        "decision_json": payload.get("decision_json", ""),
    }


def validate_strict_plan_carrier_integrity(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    export_policy = str(meta.get("plan_export_policy", "")) if isinstance(meta, dict) else ""
    if export_policy != "final_certified_accepted_faces_only":
        issues.append(
            {
                "scope": "plan_meta",
                "decision_reason": "plan_export_policy_not_final_certified_accepted_faces_only",
                "plan_export_policy": export_policy,
            }
        )
    if isinstance(meta, dict) and not bool(meta.get("strict_patchcert_carrier", False)):
        issues.append(
            {
                "scope": "plan_meta",
                "decision_reason": "plan_source_not_strict_patchcert_carrier",
                "strict_patchcert_carrier": bool(meta.get("strict_patchcert_carrier", False)),
            }
        )
    require_cluster_basis = bool(meta.get("patch_cert_cluster_basis", False)) if isinstance(meta, dict) else False
    expected_cluster_mode = str(meta.get("patch_cert_cluster_basis_mode", "")) if isinstance(meta, dict) else ""
    require_carrier_holdout = (
        bool(meta.get("patch_cert_carrier_holdout_selector", False)) if isinstance(meta, dict) else False
    )
    row_face_ids: set[int] = set()
    face_counts: dict[int, int] = {}
    for row in rows:
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            continue
        row_face_ids.add(face_id)
        face_counts[face_id] = int(face_counts.get(face_id, 0)) + 1
    for face_id, count in sorted(face_counts.items()):
        if count > 1:
            issues.append(
                {
                    "face_id": int(face_id),
                    "decision_reason": "duplicate_face_rows",
                    "row_count": int(count),
                }
            )
    patch_faces_by_face: dict[int, set[int]] = {}
    for row in rows:
        try:
            face_id = int(row.get("face_id", -1))
        except Exception:
            face_id = -1
        patch_cert = row.get("patch_certificate")
        if not isinstance(patch_cert, dict):
            issues.append({"face_id": face_id, "decision_reason": "missing_patch_certificate"})
            continue
        faces_raw = patch_cert.get("faces")
        if not isinstance(faces_raw, list) or not faces_raw:
            issues.append({"face_id": face_id, "decision_reason": "missing_patch_faces"})
            continue
        patch_faces: set[int] = set()
        for value in faces_raw:
            try:
                patch_faces.add(int(value))
            except Exception:
                continue
        if face_id not in patch_faces:
            issues.append({"face_id": face_id, "decision_reason": "patch_certificate_face_mismatch"})
        patch_faces_by_face[face_id] = patch_faces
        missing = sorted(int(fid) for fid in patch_faces if int(fid) not in row_face_ids)
        if missing:
            issues.append(
                {
                    "face_id": face_id,
                    "decision_reason": "patch_carrier_split_by_plan_rows",
                    "missing_patch_faces": missing[:20],
                    "missing_patch_face_count": int(len(missing)),
                }
            )
        cluster_basis = None
        if require_cluster_basis:
            cluster_basis = patch_cert.get("patch_cluster_basis")
            if not isinstance(cluster_basis, dict):
                issues.append({"face_id": face_id, "decision_reason": "missing_patch_cluster_basis"})
            else:
                if not bool(cluster_basis.get("enabled", False)):
                    issues.append({"face_id": face_id, "decision_reason": "patch_cluster_basis_not_enabled"})
                observed_mode = str(cluster_basis.get("mode", ""))
                if expected_cluster_mode and observed_mode != expected_cluster_mode:
                    issues.append(
                        {
                            "face_id": face_id,
                            "decision_reason": "patch_cluster_basis_mode_mismatch",
                            "expected_mode": expected_cluster_mode,
                            "observed_mode": observed_mode,
                        }
                    )
                if not bool(cluster_basis.get("passed", False)):
                    issues.append({"face_id": face_id, "decision_reason": "patch_cluster_basis_not_passed"})
                cluster_faces_raw = cluster_basis.get("faces")
                cluster_faces: set[int] = set()
                if isinstance(cluster_faces_raw, list):
                    for value in cluster_faces_raw:
                        try:
                            cluster_faces.add(int(value))
                        except Exception:
                            continue
                if cluster_faces and cluster_faces != patch_faces:
                    issues.append(
                        {
                            "face_id": face_id,
                            "decision_reason": "patch_cluster_basis_faces_mismatch",
                            "patch_face_count": int(len(patch_faces)),
                            "cluster_face_count": int(len(cluster_faces)),
                        }
                    )
                if len(patch_faces) > 1 and not bool(cluster_basis.get("applied", False)):
                    issues.append(
                        {
                            "face_id": face_id,
                            "decision_reason": "patch_cluster_basis_not_applied_for_multiface_carrier",
                            "patch_face_count": int(len(patch_faces)),
                            "cluster_rejected_reason": cluster_basis.get("rejected_reason", cluster_basis.get("skip_reason")),
                        }
                    )
        if require_carrier_holdout:
            holdout = row.get("carrier_holdout_certificate")
            if not isinstance(holdout, dict):
                issues.append({"face_id": face_id, "decision_reason": "missing_carrier_holdout_certificate"})
            elif not bool(holdout.get("passed", False)):
                issues.append({"face_id": face_id, "decision_reason": "carrier_holdout_certificate_not_passed"})
    for face_id, patch_faces in sorted(patch_faces_by_face.items()):
        for member in sorted(patch_faces):
            member_patch = patch_faces_by_face.get(int(member))
            if member_patch is None:
                continue
            if member_patch != patch_faces:
                issues.append(
                    {
                        "face_id": int(face_id),
                        "decision_reason": "inconsistent_patch_certificate_faces",
                        "other_face_id": int(member),
                    }
                )
    return issues


def plan_rows_to_facelocal_coeff(
    rows: list[dict[str, Any]],
    faces: torch.Tensor,
    *,
    fallback_basis_count: int,
    alpha_by_face: dict[int, float] | None = None,
    require_certified: bool = True,
    strict_coeff_bounds: tuple[float, float] | None = None,
) -> tuple[list[int], torch.Tensor, list[dict[str, Any]]]:
    selected_faces: list[int] = []
    coeff_rows: list[torch.Tensor] = []
    rejected: list[dict[str, Any]] = []
    face_count = int(faces.shape[0])
    for row in rows:
        face_id = int(row.get("face_id", -1))
        coeff_raw = row.get("delta_coeff", row.get("coeff"))
        reasons: list[str] = []
        if face_id < 0 or face_id >= face_count:
            reasons.append("invalid_face_id")
        if coeff_raw is None:
            reasons.append("missing_delta_coeff")
        if require_certified:
            if not bool(row.get("policy_pass", False)):
                reasons.append("policy_pass_not_true")
            if not bool(row.get("final_certified_face", False)):
                reasons.append("final_certified_face_not_true")
            patch_cert = row.get("patch_certificate")
            if not isinstance(patch_cert, dict):
                reasons.append("missing_patch_certificate")
            else:
                if bool(patch_cert.get("rejected", False)):
                    reasons.append("patch_certificate_rejected")
                if not bool(patch_cert.get("passed_patch_gain", False)):
                    reasons.append("patch_gain_not_passed")
                for key in ("patch_crossfold_certificate", "post_shrink_patch_crossfold_certificate"):
                    cert = patch_cert.get(key)
                    if not isinstance(cert, dict):
                        reasons.append(f"{key}_missing")
                    elif not bool(cert.get("enabled", False)):
                        reasons.append(f"{key}_not_enabled")
                    elif not bool(cert.get("passed", False)):
                        reasons.append(f"{key}_not_passed")
        if reasons:
            rejected.append({"face_id": face_id, "decision_reasons": reasons})
            continue
        coeff = torch.as_tensor(coeff_raw, dtype=torch.float32)
        if coeff.ndim == 2 and coeff.shape == (3, 3):
            coeff = coeff[:, None, :]
        if coeff.ndim != 3 or int(coeff.shape[0]) != 3 or int(coeff.shape[2]) != 3:
            rejected.append(
                {
                    "face_id": face_id,
                    "decision_reasons": ["invalid_delta_coeff_shape"],
                    "shape": list(coeff.shape),
                }
            )
            continue
        alpha = float((alpha_by_face or {}).get(face_id, 1.0))
        if not math.isfinite(alpha) or alpha < 0.0:
            rejected.append(
                {
                    "face_id": face_id,
                    "decision_reasons": ["invalid_materialize_plan_alpha"],
                    "alpha": alpha,
                }
            )
            continue
        coeff = coeff * alpha
        basis_count = int(coeff.shape[1])
        if basis_count <= 0:
            basis_count = int(fallback_basis_count)
        if not torch.isfinite(coeff).all():
            rejected.append(
                {
                    "face_id": face_id,
                    "decision_reasons": ["nonfinite_delta_coeff"],
                }
            )
            continue
        if strict_coeff_bounds is not None:
            max_abs_dc_coeff, max_abs_sh_coeff = strict_coeff_bounds
            if (
                not math.isfinite(float(max_abs_dc_coeff))
                or not math.isfinite(float(max_abs_sh_coeff))
                or float(max_abs_dc_coeff) <= 0.0
                or float(max_abs_sh_coeff) <= 0.0
            ):
                rejected.append(
                    {
                        "face_id": face_id,
                        "decision_reasons": ["invalid_strict_coeff_bounds"],
                        "max_abs_dc_coeff": float(max_abs_dc_coeff),
                        "max_abs_sh_coeff": float(max_abs_sh_coeff),
                    }
                )
                continue
            bounds = torch.full((basis_count,), float(max_abs_sh_coeff), dtype=torch.float32)
            bounds[0] = float(max_abs_dc_coeff)
            bound_tensor = bounds.view(1, basis_count, 1)
            excess = coeff.abs() - bound_tensor
            if bool((excess > 1e-6).any().item()):
                rejected.append(
                    {
                        "face_id": face_id,
                        "decision_reasons": ["delta_coeff_out_of_strict_bounds"],
                        "coeff_abs_max": float(coeff.abs().max().item()),
                        "max_abs_dc_coeff": float(max_abs_dc_coeff),
                        "max_abs_sh_coeff": float(max_abs_sh_coeff),
                        "max_excess": float(excess.clamp_min(0).max().item()),
                    }
                )
                continue
        selected_faces.append(face_id)
        coeff_rows.append(coeff[:, :basis_count, :])
    if not coeff_rows:
        basis_count = max(int(fallback_basis_count), 1)
        return selected_faces, torch.empty((0, basis_count, 3), dtype=torch.float32), rejected
    basis_count = max(int(row.shape[1]) for row in coeff_rows)
    padded: list[torch.Tensor] = []
    for coeff in coeff_rows:
        if int(coeff.shape[1]) == basis_count:
            padded.append(coeff)
            continue
        pad = torch.zeros((3, basis_count - int(coeff.shape[1]), 3), dtype=torch.float32)
        padded.append(torch.cat([coeff, pad], dim=1))
    return selected_faces, torch.cat(padded, dim=0), rejected


def patch_carrier_faces(patch_certificate: dict[str, Any] | None, fallback_face: int) -> list[int]:
    faces: list[int] = []
    raw_faces = patch_certificate.get("faces") if isinstance(patch_certificate, dict) else None
    if isinstance(raw_faces, list):
        for value in raw_faces:
            try:
                faces.append(int(value))
            except Exception:
                continue
    if not faces:
        faces = [int(fallback_face)]
    return sorted(set(int(face_id) for face_id in faces))


def patch_carrier_id(patch_certificate: dict[str, Any] | None, fallback_face: int) -> str:
    return "carrier_" + "_".join(str(face_id) for face_id in patch_carrier_faces(patch_certificate, fallback_face))


def patch_seed_face(patch_certificate: dict[str, Any] | None, fallback_face: int) -> int:
    if isinstance(patch_certificate, dict):
        try:
            return int(patch_certificate.get("seed_face", fallback_face))
        except Exception:
            return int(fallback_face)
    return int(fallback_face)


def patch_seed_source(
    patch_certificate: dict[str, Any] | None,
    fallback_face: int,
    strict_face_set: set[int],
    *,
    seed_rescue_enabled: bool,
) -> str:
    seed_face = patch_seed_face(patch_certificate, fallback_face)
    if seed_face in strict_face_set:
        return "strict_face_candidate_seed"
    if seed_rescue_enabled:
        return "rescued_seed"
    return "non_strict_seed"


def write_candidate_plan(
    path: Path,
    *,
    args: argparse.Namespace,
    selected_faces: list[int],
    plan_faces: list[int],
    strict_face_candidates: list[int],
    coeff: torch.Tensor,
    face_stats: dict[int, dict[str, float]],
    face_policy: dict[int, dict[str, float]],
    validation_shrink_by_face: dict[int, dict[str, Any]],
    face_view_gain_certificate: dict[int, dict[str, Any]],
    crossfold_face_gain: dict[int, dict[str, Any]],
    face_view_consensus: dict[int, dict[str, Any]],
    patch_cert_by_face: dict[int, dict[str, Any]],
    carrier_holdout_by_face: dict[int, dict[str, Any]],
    fit_proxy: dict[str, float],
    val_proxy: dict[str, float],
) -> None:
    face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
    strict_face_set = {int(fid) for fid in strict_face_candidates}
    candidates: list[dict[str, Any]] = []
    for fid in plan_faces:
        row = face_to_selected.get(int(fid))
        if row is None:
            continue
        local = coeff[row * 3 : row * 3 + 3].detach().cpu().float()
        patch_certificate = patch_cert_by_face.get(int(fid), {})
        post_cluster_proxy = patch_certificate.get("proxy", {}) if isinstance(patch_certificate, dict) else {}
        policy_proxy = post_cluster_proxy if isinstance(post_cluster_proxy, dict) and post_cluster_proxy else face_policy.get(int(fid), {})
        carrier_faces = patch_carrier_faces(patch_certificate, int(fid))
        seed_face = patch_seed_face(patch_certificate, int(fid))
        seed_source = patch_seed_source(
            patch_certificate,
            int(fid),
            strict_face_set,
            seed_rescue_enabled=bool(args.patch_cert_seed_rescue),
        )
        candidates.append(
            {
                "face_id": int(fid),
                "rank": int(len(candidates)),
                "carrier_id": patch_carrier_id(patch_certificate, int(fid)),
                "carrier_faces": carrier_faces,
                "carrier_size": int(len(carrier_faces)),
                "carrier_seed_face": int(seed_face),
                "carrier_seed_source": seed_source,
                "carrier_seed_rescued": bool(seed_source == "rescued_seed"),
                "certification_source": "seed_rescue_plus_patchcert"
                if seed_source == "rescued_seed"
                else "strict_face_seed_plus_patchcert",
                "face_was_strict_candidate": bool(int(fid) in strict_face_set),
                "delta_coeff": local.tolist(),
                "face_stats": face_stats.get(int(fid), {}),
                "policy_val_proxy": policy_proxy,
                "policy_val_proxy_source": (
                    "post_cluster_patch_certificate_proxy"
                    if isinstance(post_cluster_proxy, dict) and post_cluster_proxy
                    else "pre_cluster_face_proxy"
                ),
                "pre_cluster_policy_val_proxy": face_policy.get(int(fid), {}),
                "validation_shrink": validation_shrink_by_face.get(int(fid), {}),
                "face_view_gain_certificate": face_view_gain_certificate.get(int(fid), {}),
                "crossfold_face_gain_certificate": crossfold_face_gain.get(int(fid), {}),
                "face_view_consensus": face_view_consensus.get(int(fid), {}),
                "patch_certificate": patch_certificate,
                "post_cluster_patch_certificate": patch_certificate,
                "carrier_holdout_certificate": carrier_holdout_by_face.get(int(fid), {}),
                "post_cluster_policy_val_proxy": post_cluster_proxy,
                "policy_pass": True,
                "final_certified_face": True,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "operator": (
                    "surface_residual_facelocal_shared_field_delta_candidate_plan"
                    if bool(args.shared_residual_field)
                    else "surface_residual_facelocal_sh_delta_candidate_plan"
                ),
                "test_usage": "none",
                "source_model": str(args.source_model),
                "iteration": int(args.iteration),
                "evidence_dir": str(args.evidence_dir),
                "sh_degree": int(args.sh_degree),
                "basis_count": int((int(args.sh_degree) + 1) ** 2),
                "shared_residual_field": bool(args.shared_residual_field),
                "shared_residual_field_anchors": int(args.shared_residual_field_anchors),
                "shared_residual_field_sigma": float(args.shared_residual_field_sigma),
                "shared_residual_field_lr": float(args.shared_residual_field_lr),
                "shared_residual_field_weight_l2": float(args.shared_residual_field_weight_l2),
                "shared_residual_field_view_hinge_weight": float(args.shared_residual_field_view_hinge_weight),
                "shared_residual_field_view_hinge_min_samples": int(args.shared_residual_field_view_hinge_min_samples),
                "shared_residual_field_duplicate_smooth_weight": float(args.shared_residual_field_duplicate_smooth_weight),
                "validation_shrink_mode": str(args.validation_shrink_mode),
                "validation_gain_max_scale": float(args.validation_gain_max_scale),
                "plan_export_policy": "final_certified_accepted_faces_only",
                "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
                "patch_cert_seed_rescue": bool(args.patch_cert_seed_rescue),
                "patch_cert_seed_rescue_min_candidates": int(args.patch_cert_seed_rescue_min_candidates),
                "patch_cert_seed_rescue_max_seeds": int(args.patch_cert_seed_rescue_max_seeds),
                "patch_cert_seed_rescue_min_aux_witnesses": int(args.patch_cert_seed_rescue_min_aux_witnesses),
                "patch_cert_cluster_basis": bool(args.patch_cert_cluster_basis),
                "patch_cert_cluster_basis_mode": str(args.patch_cert_cluster_basis_mode),
                "patch_cert_cluster_basis_steps": int(args.patch_cert_cluster_basis_steps),
                "patch_cert_cluster_basis_lr": float(args.patch_cert_cluster_basis_lr),
                "patch_cert_cluster_basis_min_samples": int(args.patch_cert_cluster_basis_min_samples),
                "patch_cert_cluster_basis_max_scale": float(args.patch_cert_cluster_basis_max_scale),
                "patch_cert_cluster_basis_max_fit_mse_regression": float(args.patch_cert_cluster_basis_max_fit_mse_regression),
                "patch_cert_cluster_basis_init": str(args.patch_cert_cluster_basis_init),
                "patch_cert_cluster_basis_view_hinge_weight": float(args.patch_cert_cluster_basis_view_hinge_weight),
                "patch_cert_cluster_basis_view_hinge_min_samples": int(args.patch_cert_cluster_basis_view_hinge_min_samples),
                "patch_cert_cluster_basis_geometry_smooth_weight": float(args.patch_cert_cluster_basis_geometry_smooth_weight),
                "patch_cert_carrier_holdout_selector": bool(args.patch_cert_carrier_holdout_selector),
                "patch_cert_carrier_holdout_groups": int(args.patch_cert_carrier_holdout_groups),
                "patch_cert_carrier_holdout_grouping": str(args.patch_cert_carrier_holdout_grouping),
                "patch_cert_carrier_holdout_disjoint": bool(args.patch_cert_carrier_holdout_disjoint),
                "patch_cert_carrier_holdout_min_passing_groups": (
                    int(args.patch_cert_carrier_holdout_groups)
                    if int(args.patch_cert_carrier_holdout_min_passing_groups) <= 0
                    else int(args.patch_cert_carrier_holdout_min_passing_groups)
                ),
                "patch_cert_carrier_holdout_source": "policy_val_train_split",
                "patch_cert_carrier_holdout_min_group_relative_gain": float(
                    args.patch_cert_carrier_holdout_min_group_relative_gain
                ),
                "patch_cert_carrier_holdout_min_group_samples": int(args.patch_cert_carrier_holdout_min_group_samples),
                "patch_cert_carrier_holdout_max_mse_regression": float(args.patch_cert_carrier_holdout_max_mse_regression),
                "patch_cert_carrier_holdout_cvar_fraction": float(args.patch_cert_carrier_holdout_cvar_fraction),
                "patch_cert_carrier_holdout_cvar_weight": float(args.patch_cert_carrier_holdout_cvar_weight),
                "patch_cert_carrier_holdout_max_carriers": int(args.patch_cert_carrier_holdout_max_carriers),
                "patch_cert_carrier_holdout_auto_prefix": bool(args.patch_cert_carrier_holdout_auto_prefix),
                "patch_cert_carrier_holdout_auto_prefix_min_faces": int(
                    args.patch_cert_carrier_holdout_auto_prefix_min_faces
                ),
                "patch_cert_carrier_holdout_auto_prefix_face_bonus": float(
                    args.patch_cert_carrier_holdout_auto_prefix_face_bonus
                ),
                "patch_cert_carrier_holdout_auto_prefix_positive_tail_safe": bool(
                    args.patch_cert_carrier_holdout_auto_prefix_positive_tail_safe
                ),
                "strength": float(args.strength),
                "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
                "max_abs_dc_coeff": float(args.max_abs_delta_rgb) / float(C0),
                "max_abs_sh_coeff": (
                    float(args.max_abs_sh_coeff)
                    if float(args.max_abs_sh_coeff) > 0.0
                    else float(args.max_abs_delta_rgb) / float(C1)
                ),
                "candidate_count": int(len(candidates)),
                "carrier_count": int(len({str(row.get("carrier_id", "")) for row in candidates})),
                "fit_proxy": fit_proxy,
                "policy_val_proxy": val_proxy,
                "filters": {
                    "top_k": int(args.top_k),
                    "min_view_hits": int(args.min_view_hits),
                    "min_consistency": float(args.min_consistency),
                    "min_pixel_count": float(args.min_pixel_count),
                    "high_error_quantile": float(args.high_error_quantile),
                    "min_alpha": float(args.min_alpha),
                    "uniform_barycentric": bool(args.uniform_barycentric),
                },
                "candidates": candidates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_plan_materialize_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_facelocal_sh1_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ECSR Face-Local Surface Residual SH Delta Plan Materialization Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- plan: `{audit['materialize_plan_in']}`",
        f"- requested plan rows: `{audit['requested_plan_rows']}`",
        f"- render-trust certificate: `{audit.get('materialize_plan_render_trust', {}).get('certificate', '')}`",
        f"- render-trust accepted scale: `{audit.get('materialize_plan_render_trust', {}).get('accepted_scale', audit.get('materialize_plan_scale', 1.0))}`",
        f"- alpha json: `{audit.get('materialize_plan_alpha_json', '')}`",
        f"- alpha faces: `{audit.get('materialize_plan_alpha_faces', 0)}`",
        f"- accepted faces: `{audit['accepted_faces']}`",
        f"- vertices added: `{audit['vertices_added']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- rejected plan rows: `{len(audit['rejected_plan_rows'])}`",
        f"- triangles unchanged: `{audit['topology_triangles_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
    ]
    (output_model / "surface_residual_facelocal_sh1_delta_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_audit(output_model: Path, audit: dict[str, Any]) -> None:
    (output_model / "surface_residual_facelocal_sh1_delta_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# ECSR Face-Local Surface Residual SH Delta Audit",
        "",
        f"- operator: `{audit['operator']}`",
        f"- sh degree: `{audit['sh_degree']}`",
        f"- basis count: `{audit['basis_count']}`",
        f"- source model: `{audit['source_model']}`",
        f"- output model: `{audit['output_model']}`",
        f"- evidence dir: `{audit['evidence_dir']}`",
        f"- shared residual field: `{audit.get('shared_residual_field', False)}`",
        f"- shared residual field anchors: `{audit.get('shared_residual_field_summary', {}).get('anchor_count', 0)}`",
        f"- shared residual field parameters: `{audit.get('shared_residual_field_summary', {}).get('param_count', 0)}`",
        f"- selected faces: `{audit['selected_faces']}`",
        f"- accepted faces: `{audit['accepted_faces']}`",
        f"- vertices added: `{audit['vertices_added']}`",
        f"- fit samples: `{audit['fit_proxy']['samples']}`",
        f"- policy-val samples: `{audit['policy_val_proxy']['samples']}`",
        f"- policy-val relative gain: `{audit['policy_val_proxy']['relative_gain']:.6f}`",
        f"- carrier auto-prefix positive/tail-safe: `{audit.get('patch_cert_carrier_holdout_auto_prefix_positive_tail_safe', False)}`",
        f"- carrier auto-prefix min faces relaxed by tail safety: `{audit.get('carrier_holdout_selector', {}).get('auto_prefix_min_faces_relaxed_by_tail_safety', False)}`",
        f"- carrier auto-prefix effective min faces: `{audit.get('carrier_holdout_selector', {}).get('auto_prefix_effective_min_faces', audit.get('patch_cert_carrier_holdout_auto_prefix_min_faces', 0))}`",
        f"- validation shrink enabled: `{audit['validation_shrink']['enabled']}`",
        f"- validation shrink mode: `{audit['validation_shrink']['mode']}`",
        f"- validation shrink mean scale: `{audit['validation_shrink']['mean_scale']:.6f}`",
        f"- validation shrink zero-scale faces: `{audit['validation_shrink']['zero_scale_faces']}`",
        f"- face/view gain certificate enabled: `{audit['face_view_gain_certificate']['enabled']}`",
        f"- face/view gain certificate passing faces: `{audit['face_view_gain_certificate']['faces_passing']}`",
        f"- train-fold consistency enabled: `{audit['crossfold_face_gain_certificate']['enabled']}`",
        f"- train-fold consistency type: `{audit['crossfold_face_gain_certificate']['certificate_type']}`",
        f"- train-fold consistency passing faces: `{audit['crossfold_face_gain_certificate']['faces_passing']}`",
        f"- train-fold consistency min passing folds: `{audit['crossfold_face_gain_certificate']['min_passing_folds']}`",
        f"- face/view consensus enabled: `{audit['face_view_consensus']['enabled']}`",
        f"- face/view consensus passing faces: `{audit['face_view_consensus']['faces_passing']}`",
        f"- patch certificate enabled: `{audit['patch_certificate']['enabled']}`",
        f"- patch certificate accepted patches: `{audit['patch_certificate']['accepted_patches']}`",
        f"- patch certificate accepted faces after growth: `{audit['patch_certificate']['accepted_faces_after']}`",
        f"- accepted: `{audit['accepted']}`",
        f"- no-op copy: `{audit['no_op_copy']}`",
        f"- coeff abs mean: `{audit['coeff_abs_mean']:.8f}`",
        f"- coeff abs max: `{audit['coeff_abs_max']:.8f}`",
        f"- topology triangles unchanged: `{audit['topology_triangles_unchanged']}`",
        f"- degenerate faces: `{audit['topology_after']['degenerate_face_count']}`",
        f"- invalid indices: `{audit['topology_after']['invalid_index_count']}`",
        "",
        "This is a persistent checkpoint-level face-local appearance update. It does not read held-out test residuals.",
    ]
    (output_model / "surface_residual_facelocal_sh1_delta_audit.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if bool(args.force_apply) and args.candidate_plan_out is not None:
        raise ValueError(
            "--force_apply cannot be combined with --candidate_plan_out because forced rows "
            "do not carry the strict train-only certification required for replay."
        )
    if bool(args.patch_cert_neighbor_crossfold) and int(args.patch_cert_crossfold_folds) <= 1:
        raise ValueError(
            "--patch_cert_neighbor_crossfold requires --patch_cert_crossfold_folds > 1; "
            "otherwise neighbor admission would silently skip the fold certificate."
        )
    if bool(args.strict_patchcert_carrier):
        strict_errors: list[str] = []
        if int(args.patch_cert_rings) <= 0:
            strict_errors.append("--patch_cert_rings must be > 0")
        if int(args.patch_cert_crossfold_folds) <= 1:
            strict_errors.append("--patch_cert_crossfold_folds must be > 1")
        if int(args.patch_cert_crossfold_min_passing_folds) <= 0:
            strict_errors.append("--patch_cert_crossfold_min_passing_folds must be > 0")
        if not bool(args.patch_cert_neighbor_crossfold):
            strict_errors.append("--patch_cert_neighbor_crossfold must be enabled")
        if not bool(args.patch_cert_shrink):
            strict_errors.append("--patch_cert_shrink must be enabled")
        if not bool(args.patch_cert_carrier_holdout_selector):
            strict_errors.append("--patch_cert_carrier_holdout_selector must be enabled")
        if bool(args.force_apply):
            strict_errors.append("--force_apply is incompatible with strict PatchCert carrier mode")
        if bool(args.materialize_allow_uncertified_plan):
            strict_errors.append("--materialize_allow_uncertified_plan is incompatible with strict PatchCert carrier mode")
        if args.materialize_plan_in is not None:
            try:
                validate_strict_materialize_request(args)
            except ValueError as exc:
                strict_errors.append(str(exc))
        if strict_errors:
            raise ValueError("Strict PatchCert carrier configuration failed: " + "; ".join(strict_errors))
    source_checkpoint = checkpoint_path(args.source_model, args.iteration)
    output_checkpoint = args.output_model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(args.source_model, args.output_model)

    state = torch.load(source_checkpoint, map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu().float()

    if args.materialize_plan_in is not None:
        strict_materialize = not bool(args.materialize_allow_uncertified_plan)
        if strict_materialize:
            validate_strict_materialize_request(args)
        plan_rows, plan_meta = read_candidate_plan(
            args.materialize_plan_in,
            limit=int(args.materialize_plan_limit),
            face_ids=_parse_face_id_filter(args.materialize_plan_face_ids),
        )
        render_trust_certificate = (
            validate_render_trust_certificate(
                cert_path=args.materialize_plan_render_trust_json,
                plan_path=args.materialize_plan_in,
                requested_scale=float(args.materialize_plan_scale),
            )
            if strict_materialize
            else {"enabled": False}
        )
        plan_basis_count = int(plan_meta.get("basis_count", (int(args.sh_degree) + 1) ** 2)) if isinstance(plan_meta, dict) else int((int(args.sh_degree) + 1) ** 2)
        plan_max_abs_delta_rgb = (
            float(plan_meta.get("max_abs_delta_rgb", args.max_abs_delta_rgb))
            if isinstance(plan_meta, dict)
            else float(args.max_abs_delta_rgb)
        )
        plan_max_abs_dc_coeff = (
            float(plan_meta.get("max_abs_dc_coeff", plan_max_abs_delta_rgb / float(C0)))
            if isinstance(plan_meta, dict)
            else plan_max_abs_delta_rgb / float(C0)
        )
        plan_max_abs_sh_coeff = (
            float(
                plan_meta.get(
                    "max_abs_sh_coeff",
                    (float(args.max_abs_sh_coeff) if float(args.max_abs_sh_coeff) > 0.0 else plan_max_abs_delta_rgb / float(C1)),
                )
            )
            if isinstance(plan_meta, dict)
            else (float(args.max_abs_sh_coeff) if float(args.max_abs_sh_coeff) > 0.0 else plan_max_abs_delta_rgb / float(C1))
        )
        alpha_by_face = read_plan_alphas(args.materialize_plan_alpha_json)
        strict_plan_carrier_issues: list[dict[str, Any]] = []
        if strict_materialize:
            strict_plan_carrier_issues = validate_strict_plan_carrier_integrity(plan_rows, plan_meta)
            if strict_plan_carrier_issues:
                preview = json.dumps(strict_plan_carrier_issues[:20], indent=2)
                raise ValueError(
                    "Strict certified plan materialization rejected carrier integrity issues: "
                    + preview
                )
        accepted_faces, coeff, rejected_plan_rows = plan_rows_to_facelocal_coeff(
            plan_rows,
            faces,
            fallback_basis_count=plan_basis_count,
            alpha_by_face=alpha_by_face,
            require_certified=strict_materialize,
            strict_coeff_bounds=(plan_max_abs_dc_coeff, plan_max_abs_sh_coeff) if strict_materialize else None,
        )
        if strict_materialize and rejected_plan_rows:
            preview = json.dumps(rejected_plan_rows[:20], indent=2)
            raise ValueError(
                "Strict certified plan materialization rejected row-level certification failures: "
                + preview
            )
        coeff = coeff * float(args.materialize_plan_scale)
        if accepted_faces or not bool(args.no_op_on_fail):
            source_vertex_ids = faces[torch.as_tensor(accepted_faces, dtype=torch.long)].long().reshape(-1) if accepted_faces else torch.empty((0,), dtype=torch.long)
            out = materialize_facelocal(
                state,
                faces,
                accepted_faces,
                source_vertex_ids,
                coeff,
                accepted_faces,
            )
        else:
            out = clone_state(state)
        torch.save(out, output_checkpoint)
        degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
        topology_triangles_unchanged = int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
        vertices_added = int(out["triangles_points"].shape[0]) - int(vertices.shape[0])
        coeff_abs = coeff.abs() if coeff.numel() and accepted_faces else torch.empty((0,), dtype=torch.float32)
        no_op_copy = bool(not accepted_faces)
        accepted_set = set(int(fid) for fid in accepted_faces)
        accepted_plan_rows = [row for row in plan_rows if int(row.get("face_id", -1)) in accepted_set]
        policy_pass = bool(accepted_faces) and all(bool(row.get("policy_pass", True)) for row in accepted_plan_rows)
        audit = {
            "operator": "surface_residual_facelocal_sh_delta_plan_materialize",
            "test_usage": "none",
            "source_model": str(args.source_model),
            "source_checkpoint": str(source_checkpoint),
            "output_model": str(args.output_model),
            "output_checkpoint": str(output_checkpoint),
            "iteration": int(args.iteration),
            "materialize_plan_in": str(args.materialize_plan_in),
            "materialize_plan_limit": int(args.materialize_plan_limit),
            "materialize_plan_face_ids": str(args.materialize_plan_face_ids),
            "materialize_plan_scale": float(args.materialize_plan_scale),
            "materialize_plan_render_trust": render_trust_certificate,
            "materialize_allow_uncertified_plan": bool(args.materialize_allow_uncertified_plan),
            "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
            "strict_materialize": bool(strict_materialize),
            "strict_plan_carrier_issues": strict_plan_carrier_issues[:20],
            "materialize_plan_alpha_json": str(args.materialize_plan_alpha_json) if args.materialize_plan_alpha_json else "",
            "materialize_plan_alpha_faces": int(len(alpha_by_face)),
            "plan_source_operator": plan_meta.get("operator") if isinstance(plan_meta, dict) else None,
            "plan_export_policy": plan_meta.get("plan_export_policy") if isinstance(plan_meta, dict) else None,
            "plan_source_model": plan_meta.get("source_model") if isinstance(plan_meta, dict) else None,
            "plan_patch_cert_cluster_basis": bool(plan_meta.get("patch_cert_cluster_basis", False)) if isinstance(plan_meta, dict) else False,
            "plan_patch_cert_cluster_basis_mode": plan_meta.get("patch_cert_cluster_basis_mode") if isinstance(plan_meta, dict) else None,
            "plan_patch_cert_cluster_basis_max_scale": plan_meta.get("patch_cert_cluster_basis_max_scale") if isinstance(plan_meta, dict) else None,
            "plan_patch_cert_cluster_basis_max_fit_mse_regression": plan_meta.get("patch_cert_cluster_basis_max_fit_mse_regression") if isinstance(plan_meta, dict) else None,
            "plan_max_abs_dc_coeff": float(plan_max_abs_dc_coeff),
            "plan_max_abs_sh_coeff": float(plan_max_abs_sh_coeff),
            "requested_plan_rows": int(len(plan_rows)),
            "rejected_plan_rows": rejected_plan_rows[:20],
            "selected_faces": int(len(accepted_faces)),
            "candidate_faces": int(len(plan_rows)),
            "accepted_faces": int(len(accepted_faces)),
            "vertices_added": int(vertices_added),
            "sh_degree": int(round((plan_basis_count**0.5) - 1)) if plan_basis_count > 0 else int(args.sh_degree),
            "basis_count": int(plan_basis_count),
            "accepted": bool(accepted_faces),
            "policy_pass": bool(policy_pass),
            "force_apply": bool(args.force_apply),
            "no_op_copy": no_op_copy,
            "coeff_abs_mean": float(coeff_abs.mean().item()) if coeff_abs.numel() else 0.0,
            "coeff_abs_max": float(coeff_abs.max().item()) if coeff_abs.numel() else 0.0,
            "topology_before": {
                "triangles": int(faces.shape[0]),
                "vertices": int(vertices.shape[0]),
            },
            "topology_after": {
                "triangles": int(out["_triangle_indices"].shape[0]),
                "vertices": int(out["triangles_points"].shape[0]),
                "degenerate_face_count": int(degenerate),
                "invalid_index_count": int(invalid),
            },
            "topology_triangles_unchanged": bool(topology_triangles_unchanged),
            "accepted_preview": [
                {
                    "face_id": int(row.get("face_id", -1)),
                    "rank": int(row.get("rank", idx)),
                    "policy_val_proxy": row.get("policy_val_proxy", {}),
                    "face_stats": row.get("face_stats", {}),
                }
                for idx, row in enumerate(plan_rows[:20])
                if int(row.get("face_id", -1)) in accepted_set
            ],
        }
        write_plan_materialize_audit(args.output_model, audit)
        print(json.dumps(audit, indent=2))
        return 0 if degenerate == 0 and invalid == 0 else 1

    selected_faces, face_stats = read_selected_faces(
        args.evidence_dir / "top_residual_supports.csv",
        top_k=int(args.top_k),
        min_view_hits=int(args.min_view_hits),
        min_consistency=float(args.min_consistency),
        min_pixel_count=float(args.min_pixel_count),
    )
    selected_faces = [fid for fid in selected_faces if 0 <= int(fid) < int(faces.shape[0])]

    view_paths = sorted((args.evidence_dir / "views").glob("*.npz"))
    if not view_paths:
        view_paths = sorted((args.evidence_dir / "per_view_npz").glob("*.npz"))
    fit_paths, val_paths = split_view_paths(view_paths, int(args.policy_val_stride))
    fit_samples = collect_samples(
        fit_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=int(args.max_total_samples),
        uniform_barycentric=bool(args.uniform_barycentric),
        face_score_weight_power=float(args.face_score_weight_power),
        face_score_weight_max=float(args.face_score_weight_max),
        region_index=load_region_carrier_index(args.region_carrier_json),
        region_core_weight=float(args.region_core_weight),
        region_context_weight=float(args.region_context_weight),
        region_outside_weight=float(args.region_outside_weight),
        region_boundary_px=int(args.region_boundary_px),
    )
    val_samples = collect_samples(
        val_paths,
        selected_faces,
        face_stats,
        high_error_quantile=float(args.high_error_quantile),
        min_alpha=float(args.min_alpha),
        barycentric_tolerance=float(args.barycentric_tolerance),
        max_samples_per_face_view=int(args.max_samples_per_face_view),
        max_total_samples=max(int(args.max_total_samples // 2), 1),
        uniform_barycentric=bool(args.uniform_barycentric),
        face_score_weight_power=float(args.face_score_weight_power),
        face_score_weight_max=float(args.face_score_weight_max),
        region_index=load_region_carrier_index(args.region_carrier_json),
        region_core_weight=float(args.region_core_weight),
        region_context_weight=float(args.region_context_weight),
        region_outside_weight=float(args.region_outside_weight),
        region_boundary_px=int(args.region_boundary_px),
    )
    policy_val_all_sample_count = int(val_samples.count)
    carrier_holdout_samples: PixelSamples | None = None
    carrier_holdout_sample_count = 0
    if bool(args.patch_cert_carrier_holdout_selector) and bool(args.patch_cert_carrier_holdout_disjoint):
        if val_samples.count > 1:
            sample_indices = np.arange(int(val_samples.count), dtype=np.int64)
            tune_mask = (sample_indices % 2) == 0
            holdout_mask = ~tune_mask
            carrier_holdout_samples = subset_pixel_samples(val_samples, holdout_mask)
            val_samples = subset_pixel_samples(val_samples, tune_mask)
            carrier_holdout_sample_count = int(carrier_holdout_samples.count)
        else:
            carrier_holdout_samples = subset_pixel_samples(val_samples, np.zeros((int(val_samples.count),), dtype=bool))
            val_samples = subset_pixel_samples(val_samples, np.ones((int(val_samples.count),), dtype=bool))

    if selected_faces and fit_samples.count:
        source_vertex_ids, selected_faces_local, fit_sample_vertex_ids = localize_samples(faces, selected_faces, fit_samples)
        _, _, val_sample_vertex_ids = localize_samples(faces, selected_faces, val_samples) if val_samples.count else (
            source_vertex_ids,
            selected_faces_local,
            torch.empty((0, 3), dtype=torch.long),
        )
    else:
        source_vertex_ids = torch.empty((0,), dtype=torch.long)
        selected_faces_local = torch.empty((0, 3), dtype=torch.long)
        fit_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)
        val_sample_vertex_ids = torch.empty((0, 3), dtype=torch.long)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vertices_local = vertices[source_vertex_ids].float() if source_vertex_ids.numel() else torch.empty((0, 3), dtype=torch.float32)
    fit_ids, fit_basis, fit_target, fit_weights = samples_to_tensors(
        fit_samples,
        fit_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
        sh_degree=int(args.sh_degree),
    )
    val_ids, val_basis, val_target, val_weights = samples_to_tensors(
        val_samples,
        val_sample_vertex_ids,
        vertices_local,
        strength=float(args.strength),
        max_abs_delta_rgb=float(args.max_abs_delta_rgb),
        device=device,
        sh_degree=int(args.sh_degree),
    )

    max_abs_dc_coeff = float(args.max_abs_delta_rgb) / float(C0)
    max_abs_sh_coeff = float(args.max_abs_sh_coeff) if float(args.max_abs_sh_coeff) > 0 else float(args.max_abs_delta_rgb) / float(C1)
    if bool(args.shared_residual_field):
        coeff, solver = solve_shared_residual_field_delta(
            selected_faces_local,
            source_vertex_ids,
            vertices_local,
            fit_ids,
            fit_basis,
            fit_target,
            fit_weights,
            fit_samples.view_names,
            vertex_count=int(source_vertex_ids.shape[0]),
            max_abs_dc_coeff=max_abs_dc_coeff,
            max_abs_sh_coeff=max_abs_sh_coeff,
            lambda_mag=float(args.lambda_mag),
            lambda_sh1_mag=float(args.lambda_sh1_mag),
            lambda_smooth=float(args.lambda_smooth),
            steps=int(args.steps),
            lr=float(args.shared_residual_field_lr) if float(args.shared_residual_field_lr) > 0.0 else float(args.lr),
            anchor_count=int(args.shared_residual_field_anchors),
            sigma=float(args.shared_residual_field_sigma),
            weight_l2=float(args.shared_residual_field_weight_l2),
            view_hinge_weight=float(args.shared_residual_field_view_hinge_weight),
            view_hinge_min_samples=int(args.shared_residual_field_view_hinge_min_samples),
            duplicate_smooth_weight=float(args.shared_residual_field_duplicate_smooth_weight),
            device=device,
        )
    else:
        coeff, solver = solve_coeff_delta(
            selected_faces_local,
            fit_ids,
            fit_basis,
            fit_target,
            fit_weights,
            vertex_count=int(source_vertex_ids.shape[0]),
            max_abs_dc_coeff=max_abs_dc_coeff,
            max_abs_sh_coeff=max_abs_sh_coeff,
            lambda_mag=float(args.lambda_mag),
            lambda_sh1_mag=float(args.lambda_sh1_mag),
            lambda_smooth=float(args.lambda_smooth),
            steps=int(args.steps),
            lr=float(args.lr),
            device=device,
        )
    coeff_device = coeff.to(device=device)
    coeff_device, validation_shrink_summary, validation_shrink_by_face = calibrate_coeff_by_policy_val(
        coeff_device,
        val_ids,
        val_basis,
        val_samples,
        val_target,
        val_weights,
        selected_faces,
        mode=str(args.validation_shrink_mode),
        min_samples=int(args.validation_shrink_min_samples),
        max_gain_scale=float(args.validation_gain_max_scale),
    )
    coeff = coeff_device.detach().cpu()
    fit_proxy = evaluate_proxy(coeff_device, fit_ids, fit_basis, fit_target, fit_weights)
    val_proxy = evaluate_proxy(coeff_device, val_ids, val_basis, val_target, val_weights)
    face_policy = evaluate_proxy_by_face(coeff_device, val_ids, val_basis, val_target, val_weights, val_samples.face_ids)
    face_view_gain_certificate, face_view_gain_certificate_summary = face_view_gain_certificate_report(
        coeff_device,
        val_ids,
        val_basis,
        val_samples,
        val_target,
        val_weights,
        min_views=int(args.min_face_gain_certificate_views),
        min_relative_gain=float(args.min_face_gain_certificate_relative_gain),
        min_view_samples=int(args.min_face_gain_certificate_view_samples),
        min_fraction=float(args.min_face_gain_certificate_fraction),
    )
    crossfold_face_gain, crossfold_face_gain_summary = summarize_crossfold_face_gain(
        coeff=coeff_device,
        faces=faces,
        selected_faces=selected_faces,
        source_vertex_ids=source_vertex_ids,
        vertices=vertices,
        view_paths=view_paths,
        face_stats=face_stats,
        args=args,
        device=device,
        holdout_samples=carrier_holdout_samples,
    )
    patch_crossfold_cache, patch_crossfold_cache_summary = build_patch_crossfold_cache(
        coeff=coeff_device,
        faces=faces,
        selected_faces=selected_faces,
        source_vertex_ids=source_vertex_ids,
        vertices=vertices,
        view_paths=view_paths,
        face_stats=face_stats,
        args=args,
        device=device,
    )
    carrier_holdout_cache, carrier_holdout_cache_summary = build_carrier_holdout_cache(
        coeff=coeff_device,
        faces=faces,
        selected_faces=selected_faces,
        source_vertex_ids=source_vertex_ids,
        vertices=vertices,
        view_paths=val_paths,
        face_stats=face_stats,
        args=args,
        device=device,
        holdout_samples=carrier_holdout_samples,
    )
    face_view_consensus, face_view_consensus_summary = face_view_consensus_report(
        val_samples,
        val_target,
        val_weights,
        min_consensus=float(args.min_face_view_consensus),
        min_views=int(args.min_face_consensus_views),
        min_view_samples=int(args.min_face_consensus_view_samples),
        min_cosine=float(args.face_consensus_min_cosine),
    )
    fit_unique_faces = int(np.unique(fit_samples.face_ids).size) if fit_samples.count else 0
    val_unique_faces = int(np.unique(val_samples.face_ids).size) if val_samples.count else 0

    global_policy_pass = (
        fit_samples.count > 0
        and val_samples.count >= int(args.min_policy_val_samples)
        and val_unique_faces >= int(args.min_policy_val_unique_faces)
        and float(val_proxy["relative_gain"]) >= float(args.min_policy_val_relative_gain)
    )
    face_candidates: list[int] = []
    for fid in selected_faces:
        stats = face_policy.get(int(fid), {})
        if int(stats.get("samples", 0)) < int(args.min_face_policy_val_samples):
            continue
        if float(stats.get("relative_gain", -1.0)) < float(args.min_face_policy_val_relative_gain):
            continue
        gain_certificate = face_view_gain_certificate.get(int(fid), {})
        if bool(face_view_gain_certificate_summary.get("enabled", False)) and not bool(gain_certificate.get("passed", False)):
            continue
        crossfold_certificate = crossfold_face_gain.get(int(fid), {})
        if bool(crossfold_face_gain_summary.get("enabled", False)) and not bool(crossfold_certificate.get("passed", False)):
            continue
        consensus = face_view_consensus.get(int(fid), {})
        if bool(face_view_consensus_summary.get("enabled", False)) and not bool(consensus.get("passed", False)):
            continue
        face_candidates.append(int(fid))
    face_candidates.sort(
        key=lambda fid: (
            float(face_policy.get(fid, {}).get("relative_gain", 0.0)),
            float(face_stats.get(fid, {}).get("score", 0.0)),
            float(face_stats.get(fid, {}).get("pixel_count", 0.0)),
        ),
        reverse=True,
    )
    strict_face_candidates = list(face_candidates)
    face_candidates, patch_cert_seed_rescue_summary = apply_patch_cert_seed_rescue(
        strict_face_candidates=strict_face_candidates,
        selected_faces=selected_faces,
        face_stats=face_stats,
        face_policy=face_policy,
        face_view_gain_certificate=face_view_gain_certificate,
        face_view_gain_certificate_summary=face_view_gain_certificate_summary,
        crossfold_face_gain=crossfold_face_gain,
        crossfold_face_gain_summary=crossfold_face_gain_summary,
        face_view_consensus=face_view_consensus,
        face_view_consensus_summary=face_view_consensus_summary,
        args=args,
    )
    accepted_faces = face_candidates[: max(int(args.max_faces_to_apply), 0)]
    accepted_faces, patch_cert_summary, patch_cert_by_face = grow_patch_certified_faces(
        coeff=coeff_device,
        faces=faces,
        vertices=vertices,
        selected_faces=selected_faces,
        seed_faces=accepted_faces,
        face_stats=face_stats,
        face_policy=face_policy,
        fit_ids=fit_ids,
        fit_basis=fit_basis,
        fit_target=fit_target,
        fit_weights=fit_weights,
        fit_samples=fit_samples,
        val_ids=val_ids,
        val_basis=val_basis,
        val_target=val_target,
        val_weights=val_weights,
        val_samples=val_samples,
        patch_crossfold_cache=patch_crossfold_cache,
        args=args,
    )
    accepted_faces, carrier_holdout_summary, carrier_holdout_by_face = select_holdout_stable_carriers(
        coeff=coeff_device,
        accepted_faces=accepted_faces,
        patch_cert_by_face=patch_cert_by_face,
        holdout_cache=carrier_holdout_cache,
        args=args,
    )
    coeff = coeff_device.detach().cpu()
    accepted = bool((global_policy_pass and accepted_faces) or args.force_apply)
    if bool(args.force_apply) and not accepted_faces:
        accepted_faces = selected_faces[: max(int(args.max_faces_to_apply), 0)]
        accepted = bool(accepted_faces)
    final_accepted_fit_proxy = evaluate_proxy_for_faces(
        coeff_device,
        fit_ids,
        fit_basis,
        fit_target,
        fit_weights,
        fit_samples.face_ids,
        accepted_faces,
    )
    final_accepted_policy_val_proxy = evaluate_proxy_for_faces(
        coeff_device,
        val_ids,
        val_basis,
        val_target,
        val_weights,
        val_samples.face_ids,
        accepted_faces,
    )
    no_op_copy = bool((not accepted) and args.no_op_on_fail)

    if args.candidate_plan_out is not None:
        write_candidate_plan(
            args.candidate_plan_out,
            args=args,
            selected_faces=selected_faces,
            plan_faces=accepted_faces if accepted else [],
            strict_face_candidates=strict_face_candidates,
            coeff=coeff,
            face_stats=face_stats,
            face_policy=face_policy,
            validation_shrink_by_face=validation_shrink_by_face,
            face_view_gain_certificate=face_view_gain_certificate,
            crossfold_face_gain=crossfold_face_gain,
            face_view_consensus=face_view_consensus,
            patch_cert_by_face=patch_cert_by_face,
            carrier_holdout_by_face=carrier_holdout_by_face,
            fit_proxy=fit_proxy,
            val_proxy=val_proxy,
        )

    if accepted or not args.no_op_on_fail:
        out = materialize_facelocal(state, faces, selected_faces, source_vertex_ids, coeff, accepted_faces)
    else:
        out = clone_state(state)
    torch.save(out, output_checkpoint)

    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    topology_triangles_unchanged = int(out["_triangle_indices"].shape[0]) == int(faces.shape[0])
    vertices_added = int(out["triangles_points"].shape[0]) - int(vertices.shape[0])
    accepted_coeff_abs = torch.empty((0,), dtype=torch.float32)
    if accepted_faces and coeff.numel():
        face_to_selected = {int(fid): idx for idx, fid in enumerate(selected_faces)}
        local_ids: list[int] = []
        for fid in accepted_faces:
            row = face_to_selected[int(fid)]
            local_ids.extend([row * 3, row * 3 + 1, row * 3 + 2])
        accepted_coeff_abs = coeff[torch.as_tensor(local_ids, dtype=torch.long)].abs()
    audit = {
        "operator": (
            "surface_residual_facelocal_shared_field_delta"
            if bool(args.shared_residual_field)
            else "surface_residual_facelocal_sh_delta"
        ),
        "test_usage": "none",
        "source_model": str(args.source_model),
        "source_checkpoint": str(source_checkpoint),
        "output_model": str(args.output_model),
        "output_checkpoint": str(output_checkpoint),
        "iteration": int(args.iteration),
        "sh_degree": int(args.sh_degree),
        "basis_count": int((int(args.sh_degree) + 1) ** 2),
        "evidence_dir": str(args.evidence_dir),
        "selected_faces": int(len(selected_faces)),
        "face_policy_candidates": int(len(face_candidates)),
        "strict_face_policy_candidates": int(len(strict_face_candidates)),
        "accepted_faces": int(len(accepted_faces)) if accepted else 0,
        "vertices_added": int(vertices_added if accepted else 0),
        "fit_views": [p.stem for p in fit_paths],
        "policy_val_views": [p.stem for p in val_paths],
        "policy_val_all_samples": int(policy_val_all_sample_count),
        "policy_val_tuning_samples": int(val_samples.count),
        "carrier_holdout_disjoint_samples": int(carrier_holdout_sample_count),
        "carrier_holdout_disjoint_from_policy_tuning": bool(args.patch_cert_carrier_holdout_disjoint),
        "fit_region_bins": summarize_region_bins(fit_samples),
        "policy_val_region_bins": summarize_region_bins(val_samples),
        "fit_proxy": fit_proxy,
        "policy_val_proxy": val_proxy,
        "final_accepted_fit_proxy": final_accepted_fit_proxy,
        "final_accepted_policy_val_proxy": final_accepted_policy_val_proxy,
        "policy_val_proxy_scope": "all_selected_pre_patch_growth",
        "final_accepted_policy_val_proxy_scope": "accepted_carrier_faces_after_cluster_basis_and_shrink",
        "fit_unique_faces": int(fit_unique_faces),
        "policy_val_unique_faces": int(val_unique_faces),
        "solver": solver,
        "shared_residual_field": bool(args.shared_residual_field),
        "shared_residual_field_summary": solver.get("shared_residual_field", {}) if isinstance(solver, dict) else {},
        "validation_shrink": validation_shrink_summary,
        "face_view_gain_certificate": face_view_gain_certificate_summary,
        "crossfold_face_gain_certificate": crossfold_face_gain_summary,
        "face_view_consensus": face_view_consensus_summary,
        "patch_crossfold_cache": patch_crossfold_cache_summary,
        "patch_cert_seed_rescue": patch_cert_seed_rescue_summary,
        "patch_certificate": patch_cert_summary,
        "carrier_holdout_cache": carrier_holdout_cache_summary,
        "carrier_holdout_selector": carrier_holdout_summary,
        "filters": {
            "top_k": int(args.top_k),
            "min_view_hits": int(args.min_view_hits),
            "min_consistency": float(args.min_consistency),
            "min_pixel_count": float(args.min_pixel_count),
            "high_error_quantile": float(args.high_error_quantile),
            "min_alpha": float(args.min_alpha),
            "barycentric_tolerance": float(args.barycentric_tolerance),
            "uniform_barycentric": bool(args.uniform_barycentric),
            "face_score_weight_power": float(args.face_score_weight_power),
            "face_score_weight_max": float(args.face_score_weight_max),
            "region_carrier_json": str(args.region_carrier_json) if args.region_carrier_json is not None else "",
            "region_core_weight": float(args.region_core_weight),
            "region_context_weight": float(args.region_context_weight),
            "region_outside_weight": float(args.region_outside_weight),
            "region_boundary_px": int(args.region_boundary_px),
        },
        "strength": float(args.strength),
        "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
        "max_abs_dc_coeff": float(max_abs_dc_coeff),
        "max_abs_sh_coeff": float(max_abs_sh_coeff),
        "lambda_mag": float(args.lambda_mag),
        "lambda_sh1_mag": float(args.lambda_sh1_mag),
        "lambda_smooth": float(args.lambda_smooth),
        "shared_residual_field_anchors": int(args.shared_residual_field_anchors),
        "shared_residual_field_sigma": float(args.shared_residual_field_sigma),
        "shared_residual_field_lr": float(args.shared_residual_field_lr),
        "shared_residual_field_weight_l2": float(args.shared_residual_field_weight_l2),
        "shared_residual_field_view_hinge_weight": float(args.shared_residual_field_view_hinge_weight),
        "shared_residual_field_view_hinge_min_samples": int(args.shared_residual_field_view_hinge_min_samples),
        "shared_residual_field_duplicate_smooth_weight": float(args.shared_residual_field_duplicate_smooth_weight),
        "max_faces_to_apply": int(args.max_faces_to_apply),
        "min_policy_val_relative_gain": float(args.min_policy_val_relative_gain),
        "min_policy_val_samples": int(args.min_policy_val_samples),
        "min_policy_val_unique_faces": int(args.min_policy_val_unique_faces),
        "validation_shrink_mode": str(args.validation_shrink_mode),
        "validation_shrink_min_samples": int(args.validation_shrink_min_samples),
        "validation_gain_max_scale": float(args.validation_gain_max_scale),
        "crossfold_gain_certificate_folds": int(args.crossfold_gain_certificate_folds),
        "crossfold_min_passing_folds": int(args.crossfold_min_passing_folds),
        "crossfold_min_fold_relative_gain": float(args.crossfold_min_fold_relative_gain),
        "crossfold_min_fold_samples": int(args.crossfold_min_fold_samples),
        "min_face_policy_val_relative_gain": float(args.min_face_policy_val_relative_gain),
        "min_face_policy_val_samples": int(args.min_face_policy_val_samples),
        "min_face_gain_certificate_views": int(args.min_face_gain_certificate_views),
        "min_face_gain_certificate_relative_gain": float(args.min_face_gain_certificate_relative_gain),
        "min_face_gain_certificate_view_samples": int(args.min_face_gain_certificate_view_samples),
        "min_face_gain_certificate_fraction": float(args.min_face_gain_certificate_fraction),
        "min_face_view_consensus": float(args.min_face_view_consensus),
        "min_face_consensus_views": int(args.min_face_consensus_views),
        "min_face_consensus_view_samples": int(args.min_face_consensus_view_samples),
        "face_consensus_min_cosine": float(args.face_consensus_min_cosine),
        "patch_cert_rings": int(args.patch_cert_rings),
        "patch_cert_max_faces_per_seed": int(args.patch_cert_max_faces_per_seed),
        "patch_cert_min_direction_cosine": float(args.patch_cert_min_direction_cosine),
        "patch_cert_min_neighbor_policy_val_samples": int(args.patch_cert_min_neighbor_policy_val_samples),
        "patch_cert_min_neighbor_policy_val_relative_gain": float(args.patch_cert_min_neighbor_policy_val_relative_gain),
        "patch_cert_min_policy_val_samples": int(args.patch_cert_min_policy_val_samples),
        "patch_cert_min_relative_gain": float(args.patch_cert_min_relative_gain),
        "patch_cert_neighbor_mode": str(args.patch_cert_neighbor_mode),
        "patch_cert_centroid_candidates_per_seed": int(args.patch_cert_centroid_candidates_per_seed),
        "patch_cert_seed_rescue_enabled": bool(args.patch_cert_seed_rescue),
        "patch_cert_seed_rescue_min_candidates": int(args.patch_cert_seed_rescue_min_candidates),
        "patch_cert_seed_rescue_max_seeds": int(args.patch_cert_seed_rescue_max_seeds),
        "patch_cert_seed_rescue_min_aux_witnesses": int(args.patch_cert_seed_rescue_min_aux_witnesses),
        "patch_cert_crossfold_folds": int(args.patch_cert_crossfold_folds),
        "patch_cert_crossfold_min_passing_folds": int(args.patch_cert_crossfold_min_passing_folds),
        "patch_cert_crossfold_min_fold_relative_gain": float(args.patch_cert_crossfold_min_fold_relative_gain),
        "patch_cert_crossfold_min_fold_samples": int(args.patch_cert_crossfold_min_fold_samples),
        "patch_cert_neighbor_crossfold": bool(args.patch_cert_neighbor_crossfold),
        "patch_cert_shrink": bool(args.patch_cert_shrink),
        "patch_cert_cluster_basis": bool(args.patch_cert_cluster_basis),
        "patch_cert_cluster_basis_mode": str(args.patch_cert_cluster_basis_mode),
        "patch_cert_cluster_basis_steps": int(args.patch_cert_cluster_basis_steps),
        "patch_cert_cluster_basis_lr": float(args.patch_cert_cluster_basis_lr),
        "patch_cert_cluster_basis_min_samples": int(args.patch_cert_cluster_basis_min_samples),
        "patch_cert_cluster_basis_max_scale": float(args.patch_cert_cluster_basis_max_scale),
        "patch_cert_cluster_basis_max_fit_mse_regression": float(args.patch_cert_cluster_basis_max_fit_mse_regression),
        "patch_cert_cluster_basis_init": str(args.patch_cert_cluster_basis_init),
        "patch_cert_cluster_basis_view_hinge_weight": float(args.patch_cert_cluster_basis_view_hinge_weight),
        "patch_cert_cluster_basis_view_hinge_min_samples": int(args.patch_cert_cluster_basis_view_hinge_min_samples),
        "patch_cert_cluster_basis_geometry_smooth_weight": float(args.patch_cert_cluster_basis_geometry_smooth_weight),
        "patch_cert_carrier_holdout_selector": bool(args.patch_cert_carrier_holdout_selector),
        "patch_cert_carrier_holdout_groups": int(args.patch_cert_carrier_holdout_groups),
        "patch_cert_carrier_holdout_grouping": str(args.patch_cert_carrier_holdout_grouping),
        "patch_cert_carrier_holdout_disjoint": bool(args.patch_cert_carrier_holdout_disjoint),
        "patch_cert_carrier_holdout_min_passing_groups": (
            int(args.patch_cert_carrier_holdout_groups)
            if int(args.patch_cert_carrier_holdout_min_passing_groups) <= 0
            else int(args.patch_cert_carrier_holdout_min_passing_groups)
        ),
        "patch_cert_carrier_holdout_source": "policy_val_train_split",
        "patch_cert_carrier_holdout_min_group_relative_gain": float(args.patch_cert_carrier_holdout_min_group_relative_gain),
        "patch_cert_carrier_holdout_min_group_samples": int(args.patch_cert_carrier_holdout_min_group_samples),
        "patch_cert_carrier_holdout_max_mse_regression": float(args.patch_cert_carrier_holdout_max_mse_regression),
        "patch_cert_carrier_holdout_cvar_fraction": float(args.patch_cert_carrier_holdout_cvar_fraction),
        "patch_cert_carrier_holdout_cvar_weight": float(args.patch_cert_carrier_holdout_cvar_weight),
        "patch_cert_carrier_holdout_max_carriers": int(args.patch_cert_carrier_holdout_max_carriers),
        "patch_cert_carrier_holdout_auto_prefix": bool(args.patch_cert_carrier_holdout_auto_prefix),
        "patch_cert_carrier_holdout_auto_prefix_min_faces": int(args.patch_cert_carrier_holdout_auto_prefix_min_faces),
        "patch_cert_carrier_holdout_auto_prefix_face_bonus": float(args.patch_cert_carrier_holdout_auto_prefix_face_bonus),
        "patch_cert_carrier_holdout_auto_prefix_positive_tail_safe": bool(
            args.patch_cert_carrier_holdout_auto_prefix_positive_tail_safe
        ),
        "strict_patchcert_carrier": bool(args.strict_patchcert_carrier),
        "global_policy_pass": bool(global_policy_pass),
        "policy_pass": bool(global_policy_pass),
        "accepted": bool(accepted),
        "force_apply": bool(args.force_apply),
        "no_op_copy": no_op_copy,
        "coeff_abs_mean": float(accepted_coeff_abs.mean().item()) if accepted_coeff_abs.numel() and accepted else 0.0,
        "coeff_abs_max": float(accepted_coeff_abs.max().item()) if accepted_coeff_abs.numel() and accepted else 0.0,
        "topology_before": {
            "triangles": int(faces.shape[0]),
            "vertices": int(vertices.shape[0]),
        },
        "topology_after": {
            "triangles": int(out["_triangle_indices"].shape[0]),
            "vertices": int(out["triangles_points"].shape[0]),
            "degenerate_face_count": int(degenerate),
            "invalid_index_count": int(invalid),
        },
        "topology_triangles_unchanged": bool(topology_triangles_unchanged),
        "accepted_preview": [
            {
                "face_id": int(fid),
                "face_stats": face_stats.get(int(fid), {}),
                "policy_val_proxy": face_policy.get(int(fid), {}),
                "validation_shrink": validation_shrink_by_face.get(int(fid), {}),
                "face_view_gain_certificate": face_view_gain_certificate.get(int(fid), {}),
                "crossfold_face_gain_certificate": crossfold_face_gain.get(int(fid), {}),
                "face_view_consensus": face_view_consensus.get(int(fid), {}),
                "carrier_id": patch_carrier_id(patch_cert_by_face.get(int(fid), {}), int(fid)),
                "carrier_faces": patch_carrier_faces(patch_cert_by_face.get(int(fid), {}), int(fid)),
                "carrier_seed_face": patch_seed_face(patch_cert_by_face.get(int(fid), {}), int(fid)),
                "carrier_seed_source": patch_seed_source(
                    patch_cert_by_face.get(int(fid), {}),
                    int(fid),
                    {int(face_id) for face_id in strict_face_candidates},
                    seed_rescue_enabled=bool(args.patch_cert_seed_rescue),
                ),
                "carrier_seed_rescued": bool(
                    patch_seed_source(
                        patch_cert_by_face.get(int(fid), {}),
                        int(fid),
                        {int(face_id) for face_id in strict_face_candidates},
                        seed_rescue_enabled=bool(args.patch_cert_seed_rescue),
                    )
                    == "rescued_seed"
                ),
                "face_was_strict_candidate": bool(int(fid) in {int(face_id) for face_id in strict_face_candidates}),
                "patch_certificate": patch_cert_by_face.get(int(fid), {}),
                "carrier_holdout_certificate": carrier_holdout_by_face.get(int(fid), {}),
            }
            for fid in (accepted_faces[:20] if accepted else [])
        ],
    }
    write_audit(args.output_model, audit)
    print(json.dumps(audit, indent=2))
    return 0 if degenerate == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
