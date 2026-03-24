import os
import re
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_MASK_PATH_CACHE = {}
_MASK_BINARY_CACHE = {}
_WARNED_MESSAGES = set()
_DEBUG_SAVED_COUNT = 0
_MASK_INDEX_CACHE = {}


def _warn_once(message: str):
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(message)


def _resolve_mask_root(source_path: str, mask_dir: str) -> Optional[str]:
    if not mask_dir:
        return None
    if os.path.isabs(mask_dir):
        return mask_dir
    return os.path.join(source_path, mask_dir)


def _extract_numeric_suffix(name: str) -> Optional[str]:
    match = re.search(r"(\d+)$", name)
    if match is None:
        return None
    return match.group(1)


def _candidate_mask_stems(image_name: str, image_path: str) -> list:
    candidates = []
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    if image_name:
        candidates.append(image_name)
        if image_name.startswith("images_"):
            candidates.append(image_name[len("images_"):])
        numeric = _extract_numeric_suffix(image_name)
        if numeric is not None:
            candidates.append(numeric)

    candidates.append(image_stem)
    if image_stem.startswith("images_"):
        candidates.append(image_stem[len("images_"):])
    numeric = _extract_numeric_suffix(image_stem)
    if numeric is not None:
        candidates.append(numeric)

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def resolve_ground_mask_path(
    source_path: str,
    mask_dir: str,
    image_name: str,
    image_path: str,
    matching: str = "auto",
    suffix: str = ".png",
    missing_strategy: str = "empty",
    nearest_max_gap: int = 6,
) -> Optional[str]:
    mask_root = _resolve_mask_root(source_path, mask_dir)
    if mask_root is None:
        return None

    cache_key = (mask_root, image_name, image_path, matching, suffix, missing_strategy, int(nearest_max_gap))
    if cache_key in _MASK_PATH_CACHE:
        return _MASK_PATH_CACHE[cache_key]

    if not os.path.isdir(mask_root):
        _warn_once(f"[GroundMask] Directory not found: {mask_root}. Ground-mask loading is skipped.")
        _MASK_PATH_CACHE[cache_key] = None
        return None

    strategy = (matching or "auto").lower()
    stems = _candidate_mask_stems(image_name=image_name, image_path=image_path)

    ordered_stems = stems
    if strategy == "exact":
        ordered_stems = stems[:1]
    elif strategy == "strip_prefix":
        ordered_stems = [s for s in stems if not s.startswith("images_")]
        if not ordered_stems:
            ordered_stems = stems
    elif strategy == "numeric":
        ordered_stems = [s for s in stems if s.isdigit()]
    elif strategy != "auto":
        _warn_once(f"[GroundMask] Unknown matching strategy '{matching}'. Falling back to auto.")

    resolved = None
    for stem in ordered_stems:
        candidate = os.path.join(mask_root, stem + suffix)
        if os.path.exists(candidate):
            resolved = candidate
            break

    if resolved is None and (missing_strategy or "empty").lower() == "nearest":
        # Optional fallback: pick nearest numeric-id mask when exact match is absent.
        query_id = None
        for s in stems:
            if s.isdigit():
                query_id = int(s)
                break
            numeric = _extract_numeric_suffix(s)
            if numeric is not None:
                query_id = int(numeric)
                break
        if query_id is not None:
            index_key = (mask_root, suffix)
            indexed = _MASK_INDEX_CACHE.get(index_key, None)
            if indexed is None:
                mapping = {}
                try:
                    for fn in os.listdir(mask_root):
                        if not fn.endswith(suffix):
                            continue
                        stem = os.path.splitext(fn)[0]
                        if stem.isdigit():
                            mapping[int(stem)] = os.path.join(mask_root, fn)
                except Exception:
                    mapping = {}
                indexed = sorted(mapping.items(), key=lambda x: x[0])
                _MASK_INDEX_CACHE[index_key] = indexed
            if len(indexed) > 0:
                ids = np.array([k for k, _ in indexed], dtype=np.int64)
                pos = int(np.argmin(np.abs(ids - query_id)))
                nearest_id = int(ids[pos])
                gap = abs(nearest_id - int(query_id))
                if gap <= int(max(0, nearest_max_gap)):
                    resolved = indexed[pos][1]
                    _warn_once(
                        f"[GroundMask] Missing exact mask for '{image_name}', using nearest id {nearest_id} (gap={gap})."
                    )

    _MASK_PATH_CACHE[cache_key] = resolved
    return resolved


def _binary_mask_from_array(
    arr: np.ndarray,
    threshold: int = 127,
    label_value: int = -1,
    label_rgb: str = "",
) -> torch.Tensor:
    if arr.ndim == 2:
        if label_value >= 0:
            mask = (arr == label_value)
        else:
            mask = (arr > threshold)
        return torch.from_numpy(mask.astype(np.bool_))

    if arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape {arr.shape}. Expected HxW or HxWxC.")

    rgb = arr[..., :3]
    if label_rgb:
        parts = [p.strip() for p in label_rgb.split(",")]
        if len(parts) != 3:
            raise ValueError("ground_mask_label_rgb must be 'R,G,B'.")
        color = np.array([int(parts[0]), int(parts[1]), int(parts[2])], dtype=np.uint8)
        mask = np.all(rgb == color[None, None, :], axis=-1)
        return torch.from_numpy(mask.astype(np.bool_))

    if label_value >= 0:
        # For RGB masks, match if any channel equals the label value.
        mask = np.any(rgb == label_value, axis=-1)
        return torch.from_numpy(mask.astype(np.bool_))

    flat_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    if flat_colors.shape[0] <= 2 and np.any(np.all(flat_colors == np.array([0, 0, 0], dtype=np.uint8), axis=1)):
        # Typical semantic binary RGB mask with black background.
        mask = np.any(rgb > 0, axis=-1)
        return torch.from_numpy(mask.astype(np.bool_))

    _warn_once(
        "[GroundMask] Non-binary RGB mask detected without explicit label. "
        "Falling back to grayscale thresholding."
    )
    gray = rgb.astype(np.float32).mean(axis=-1)
    mask = gray > float(threshold)
    return torch.from_numpy(mask.astype(np.bool_))


def load_ground_mask_binary(
    mask_path: str,
    threshold: int = 127,
    label_value: int = -1,
    label_rgb: str = "",
) -> torch.Tensor:
    cache_key = (mask_path, int(threshold), int(label_value), label_rgb)
    if cache_key in _MASK_BINARY_CACHE:
        return _MASK_BINARY_CACHE[cache_key]

    mask_img = Image.open(mask_path)
    arr = np.array(mask_img)
    binary_mask = _binary_mask_from_array(
        arr=arr,
        threshold=threshold,
        label_value=label_value,
        label_rgb=label_rgb,
    )
    _MASK_BINARY_CACHE[cache_key] = binary_mask
    return binary_mask


def resize_ground_mask(mask_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if mask_hw.dtype != torch.bool:
        mask_hw = mask_hw.bool()

    mask_f = mask_hw.float().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_f, size=(out_h, out_w), mode="nearest")
    return (mask_resized.squeeze(0).squeeze(0) > 0.5)


def maybe_save_mask_overlay(
    rgb_chw: torch.Tensor,
    mask_hw: torch.Tensor,
    image_name: str,
    debug_enabled: bool,
    debug_dir: str,
    max_examples: int,
    alpha: float = 0.45,
):
    global _DEBUG_SAVED_COUNT
    if not debug_enabled:
        return
    if _DEBUG_SAVED_COUNT >= max_examples:
        return
    if not debug_dir:
        return

    os.makedirs(debug_dir, exist_ok=True)

    rgb = rgb_chw.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    mask = mask_hw.detach().cpu().numpy().astype(np.float32)

    # Green overlay for ground pixels.
    overlay = rgb.copy()
    green = np.zeros_like(rgb)
    green[..., 1] = 1.0
    overlay = rgb * (1.0 - alpha * mask[..., None]) + green * (alpha * mask[..., None])

    out = np.concatenate([rgb, overlay], axis=1)
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_path = os.path.join(debug_dir, f"{image_name}_ground_overlay.png")
    Image.fromarray(out_u8).save(out_path)
    _DEBUG_SAVED_COUNT += 1
