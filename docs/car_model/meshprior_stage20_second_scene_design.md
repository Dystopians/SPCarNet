# MeshPrior Stage 20 Second Scene Design

Date: 2026-05-01

## Goal

Add a second real scene so MeshPrior results are not based only on `parking_phone_tiny`.

## Required Dataset Contract

A usable second scene should provide:

- an `images/` directory with ordered RGB images;
- a COLMAP sparse reconstruction with `cameras`, `images`, and `points3D` in binary or text format;
- camera/image names that match the image files;
- enough overlapping views for a train/test split;
- preferably a parking-lot or vehicle-rich scene;
- optional segmentation masks or object labels for car/ground filtering;
- notes on scale, orientation, and whether images are anonymized.

The preferred structure is:

```text
<scene_root>/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

or an equivalent COLMAP layout that can be symlinked into a repo-local dataset view.

## Gate

`PASS` if a second scene can be converted to a repo-local dataset view and baseline-smoked.

`SOFT PASS` if only an audit is possible.

`STOP` if no usable second scene exists locally.
