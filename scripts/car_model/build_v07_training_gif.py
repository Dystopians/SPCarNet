"""Build a training-progression GIF for a single patch across v0.7 train steps.

Collects the comparison panels emitted every 2000 steps under wandb media dirs,
sorts by step, resizes, annotates with step number, and dumps an animated GIF.
Pure PIL, no GPU.
"""

import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PATCH = "02c8cd6565251f264276b761f81e5c812c5937bddf59b3cce8e6a018979c3efa"

RUN_DIRS = [
    Path("/data2/peilincai/mesh-splatting/outputs/carnet/v0_7/full/wandb/"
         "run-20260423_151318-j92gorrq/files/media/images/viz_main_step/comparison"),
    Path("/data2/peilincai/mesh-splatting/outputs/carnet/v0_7/full/wandb/"
         "run-20260423_193151-kug6k293/files/media/images/viz_main_step/comparison"),
]

OUT_PATH = Path("/data2/peilincai/mesh-splatting/outputs/carnet/v0_7/eval/"
                f"training_progress_{PATCH[:12]}.gif")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_W = 1200
FRAME_MS = 600
LOOP_PAUSE_MS = 1500

STEP_RE = re.compile(rf"^{PATCH}_(\d+)_[0-9a-f]+\.png$")


def collect() -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for d in RUN_DIRS:
        if not d.exists():
            continue
        for p in d.iterdir():
            m = STEP_RE.match(p.name)
            if m:
                items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return items


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_frame(src: Path, step: int, total_steps: int) -> Image.Image:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = TARGET_W / w
    img = img.resize((TARGET_W, int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = load_font(36)
    label = f"step = {step:>6d}   ({step / total_steps * 100:4.1f}% of training)"
    # semi-transparent black box
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 12
    box_w = bbox[2] - bbox[0] + 2 * pad
    box_h = bbox[3] - bbox[1] + 2 * pad
    box = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 180))
    img.paste(box, (20, 20), box)
    draw.text((20 + pad, 20 + pad - bbox[1]), label, fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    items = collect()
    if not items:
        raise SystemExit("no images found")
    max_step = items[-1][0]
    print(f"[gif] {len(items)} frames, steps {items[0][0]} -> {max_step}")

    frames = [render_frame(p, step, max_step) for step, p in items]
    durations = [FRAME_MS] * len(frames)
    durations[-1] = LOOP_PAUSE_MS  # hold on the final frame before loop

    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"[gif] wrote {OUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
