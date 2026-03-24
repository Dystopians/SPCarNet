import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    # Professional road-scene model (recommended default).
    "segformer_cityscapes": {
        "model_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "ground_keywords": "road,sidewalk,terrain,ground,pavement,asphalt",
    },
    # Generic scene parsing baseline.
    "segformer_ade": {
        "model_name": "nvidia/segformer-b5-finetuned-ade-640-640",
        "ground_keywords": "road,sidewalk,pavement,asphalt,ground,earth,floor,dirt,path",
    },
}


def _list_images(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for fn in sorted(os.listdir(image_dir)):
        if os.path.splitext(fn.lower())[1] in exts:
            files.append(fn)
    return files


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_keywords(raw: str) -> List[str]:
    items = [x.strip().lower() for x in raw.split(",")]
    return [x for x in items if x]


def _build_ground_label_ids(id2label: dict, keywords: List[str]) -> List[int]:
    ids = []
    for k, v in id2label.items():
        label = str(v).lower()
        if any(kw in label for kw in keywords):
            ids.append(int(k))
    return sorted(set(ids))


def _postprocess_mask(mask_u8: np.ndarray, kernel_size: int, min_area: int) -> np.ndarray:
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        cleaned = np.zeros_like(mask_u8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == i] = 255
        mask_u8 = cleaned
    return mask_u8


def _save_overlay(image_np: np.ndarray, mask_u8: np.ndarray, out_path: str):
    overlay = image_np.copy()
    mask_bool = mask_u8 > 0
    overlay[mask_bool] = (
        0.45 * overlay[mask_bool].astype(np.float32)
        + 0.55 * np.array([40, 220, 40], dtype=np.float32)
    ).astype(np.uint8)
    panel = np.concatenate([image_np, overlay], axis=1)
    Image.fromarray(panel).save(out_path)


def _slugify_model_name(name: str) -> str:
    slug = name.strip().lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_\\-\\.]+", "_", slug)
    return slug


def _resolve_model_and_keywords(args) -> Tuple[str, str]:
    if args.model_name:
        model_name = args.model_name
        if args.ground_keywords:
            return model_name, args.ground_keywords
        if args.model_preset and args.model_preset in MODEL_PRESETS:
            return model_name, MODEL_PRESETS[args.model_preset]["ground_keywords"]
        return model_name, MODEL_PRESETS["segformer_cityscapes"]["ground_keywords"]

    preset = args.model_preset if args.model_preset else "segformer_cityscapes"
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown model_preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    model_name = MODEL_PRESETS[preset]["model_name"]
    keywords = args.ground_keywords if args.ground_keywords else MODEL_PRESETS[preset]["ground_keywords"]
    return model_name, keywords


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground masks with professional semantic models and model-tagged output management."
    )
    parser.add_argument("--image_dir", required=True, type=str, help="Input image directory.")
    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="Root output directory. Final output is organized under a model-tagged subdirectory.",
    )
    parser.add_argument(
        "--model_preset",
        type=str,
        default="segformer_cityscapes",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset for ground segmentation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Optional custom HF model. Overrides preset model_name when provided.",
    )
    parser.add_argument(
        "--ground_keywords",
        type=str,
        default="",
        help="Optional comma-separated keywords for selecting ground labels from id2label.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--morph_kernel", type=int, default=5, help="Morphology kernel size.")
    parser.add_argument("--min_component_area", type=int, default=200, help="Remove tiny components.")
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--overlay_max", type=int, default=80)
    parser.add_argument("--report_name", type=str, default="ground_mask_generation_report.json")
    args = parser.parse_args()

    model_name, kw_raw = _resolve_model_and_keywords(args)
    model_tag = _slugify_model_name(model_name)
    out_dir = os.path.join(args.output_root, f"ground_masks__{model_tag}")
    mask_dir = os.path.join(out_dir, "masks")
    overlay_dir = os.path.join(out_dir, "overlays")
    report_dir = os.path.join(out_dir, "reports")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except Exception as exc:
        raise RuntimeError(
            "transformers is required. Install in env: pip install transformers"
        ) from exc

    device = _resolve_device(args.device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    keywords = _parse_keywords(kw_raw)
    id2label = model.config.id2label
    ground_label_ids = _build_ground_label_ids(id2label, keywords)
    if len(ground_label_ids) == 0:
        raise RuntimeError(
            f"No ground-like labels matched keywords={keywords}. "
            "Adjust --ground_keywords for this model."
        )

    image_files = _list_images(args.image_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    report = {
        "image_dir": args.image_dir,
        "output_root": args.output_root,
        "output_dir": out_dir,
        "mask_dir": mask_dir,
        "model_preset": args.model_preset,
        "model_name": model_name,
        "model_tag": model_tag,
        "device": str(device),
        "keywords": keywords,
        "ground_label_ids": ground_label_ids,
        "ground_label_names": [id2label[int(i)] for i in ground_label_ids],
        "images_total": len(image_files),
        "items": [],
    }

    overlay_saved = 0
    with torch.no_grad():
        for start in range(0, len(image_files), max(args.batch_size, 1)):
            batch_files = image_files[start : start + max(args.batch_size, 1)]
            batch_imgs = []
            batch_sizes = []
            for fn in batch_files:
                p = os.path.join(args.image_dir, fn)
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                batch_sizes.append((img.height, img.width))

            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits

            for i, fn in enumerate(batch_files):
                h, w = batch_sizes[i]
                logit_i = logits[i : i + 1]
                logit_i = torch.nn.functional.interpolate(
                    logit_i, size=(h, w), mode="bilinear", align_corners=False
                )
                pred = logit_i.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
                mask = np.isin(pred, np.array(ground_label_ids, dtype=np.int32)).astype(np.uint8) * 255
                mask = _postprocess_mask(
                    mask,
                    kernel_size=max(int(args.morph_kernel), 1),
                    min_area=max(int(args.min_component_area), 0),
                )

                stem = os.path.splitext(fn)[0]
                out_path = os.path.join(mask_dir, stem + ".png")
                Image.fromarray(mask).save(out_path)

                ratio = float((mask > 0).mean())
                report["items"].append(
                    {
                        "image": fn,
                        "mask": os.path.relpath(out_path, out_dir),
                        "ground_ratio": ratio,
                    }
                )

                if args.save_overlay and overlay_saved < int(args.overlay_max):
                    img_np = np.array(batch_imgs[i], dtype=np.uint8)
                    _save_overlay(img_np, mask, os.path.join(overlay_dir, stem + "_overlay.png"))
                    overlay_saved += 1

    ratios = [x["ground_ratio"] for x in report["items"]]
    report["ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
    report["ratio_median"] = float(np.median(ratios)) if ratios else 0.0
    report["ratio_q05"] = float(np.quantile(ratios, 0.05)) if ratios else 0.0
    report["ratio_q95"] = float(np.quantile(ratios, 0.95)) if ratios else 0.0

    report_path = os.path.join(report_dir, args.report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[GroundMaskGenPro] Done.")
    print(f"[GroundMaskGenPro] model={model_name}")
    print(f"[GroundMaskGenPro] output_dir={out_dir}")
    print(f"[GroundMaskGenPro] mask_dir={mask_dir}")
    print(f"[GroundMaskGenPro] report={report_path}")
    print(
        "[GroundMaskGenPro] ratio mean={:.4f} median={:.4f} q05={:.4f} q95={:.4f}".format(
            report["ratio_mean"],
            report["ratio_median"],
            report["ratio_q05"],
            report["ratio_q95"],
        )
    )


if __name__ == "__main__":
    main()
