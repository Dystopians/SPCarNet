import argparse
import json
import os
from typing import List

import cv2
import numpy as np
from PIL import Image
import torch


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


def main():
    parser = argparse.ArgumentParser(description="Generate per-image ground masks using semantic segmentation.")
    parser.add_argument("--image_dir", required=True, type=str, help="Input image directory.")
    parser.add_argument("--output_dir", required=True, type=str, help="Output binary mask directory.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/segformer-b5-finetuned-ade-640-640",
        help="HF semantic segmentation model.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--ground_keywords",
        type=str,
        default="road,sidewalk,pavement,asphalt,ground,earth,floor,dirt,path",
        help="Comma-separated label keywords to be treated as ground.",
    )
    parser.add_argument("--morph_kernel", type=int, default=5, help="Morphology kernel size.")
    parser.add_argument("--min_component_area", type=int, default=200, help="Remove tiny components.")
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--overlay_max", type=int, default=40)
    parser.add_argument("--report_json", type=str, default="ground_mask_generation_report.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "_overlay")
    if args.save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for generate_ground_masks.py. "
            "Install it in your env, e.g. `pip install transformers`."
        ) from exc

    device = _resolve_device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name).to(device)
    model.eval()

    keywords = _parse_keywords(args.ground_keywords)
    id2label = model.config.id2label
    ground_label_ids = _build_ground_label_ids(id2label, keywords)
    if len(ground_label_ids) == 0:
        raise RuntimeError(
            f"No ground-like labels matched keywords={keywords}. "
            "Please adjust --ground_keywords for this model's label space."
        )

    image_files = _list_images(args.image_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    report = {
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
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
            logits = outputs.logits  # [B, C, h, w]

            for i, fn in enumerate(batch_files):
                h, w = batch_sizes[i]
                logit_i = logits[i : i + 1]
                logit_i = torch.nn.functional.interpolate(
                    logit_i, size=(h, w), mode="bilinear", align_corners=False
                )
                pred = logit_i.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
                mask = np.isin(pred, np.array(ground_label_ids, dtype=np.int32)).astype(np.uint8) * 255
                mask = _postprocess_mask(mask, kernel_size=max(args.morph_kernel, 1), min_area=max(args.min_component_area, 0))

                stem = os.path.splitext(fn)[0]
                out_path = os.path.join(args.output_dir, stem + ".png")
                Image.fromarray(mask).save(out_path)

                ratio = float((mask > 0).mean())
                report["items"].append({"image": fn, "mask": os.path.basename(out_path), "ground_ratio": ratio})

                if args.save_overlay and overlay_saved < int(args.overlay_max):
                    img_np = np.array(batch_imgs[i], dtype=np.uint8)
                    _save_overlay(img_np, mask, os.path.join(overlay_dir, stem + "_overlay.png"))
                    overlay_saved += 1

    ratios = [x["ground_ratio"] for x in report["items"]]
    report["ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
    report["ratio_median"] = float(np.median(ratios)) if ratios else 0.0
    report["ratio_q05"] = float(np.quantile(ratios, 0.05)) if ratios else 0.0
    report["ratio_q95"] = float(np.quantile(ratios, 0.95)) if ratios else 0.0

    report_path = os.path.join(args.output_dir, args.report_json)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[GroundMaskGen] Done.")
    print(f"[GroundMaskGen] images={len(image_files)} output_dir={args.output_dir}")
    print(f"[GroundMaskGen] labels={report['ground_label_names']}")
    print(
        "[GroundMaskGen] ratio mean={:.4f} median={:.4f} q05={:.4f} q95={:.4f}".format(
            report["ratio_mean"],
            report["ratio_median"],
            report["ratio_q05"],
            report["ratio_q95"],
        )
    )
    print(f"[GroundMaskGen] report={report_path}")


if __name__ == "__main__":
    main()
