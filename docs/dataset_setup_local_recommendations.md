# Local Dataset Setup (Indoor + Outdoor)

This note tracks datasets imported locally for MeshSplatting training/evaluation in this workspace.

## Imported datasets

Base directory:

`/data2/peilincai/mesh_datasets/mipnerf360`

Included scenes:

- Indoor: `bonsai`
- Outdoor: `flowers`

## Source and compatibility

Both scenes were downloaded from the official MipNeRF-360 `refraw360` release:

- `http://storage.googleapis.com/gresearch/refraw360/bonsai.zip`
- `http://storage.googleapis.com/gresearch/refraw360/flowers.zip`

These scenes are directly compatible with this repo's COLMAP loader (`scene/dataset_readers.py`) and include:

- `images`, `images_2`, `images_4`, `images_8`
- `sparse/0/cameras.bin`
- `sparse/0/images.bin`
- `sparse/0/points3D.bin`

## Recommended commands

Train indoor:

```bash
python train.py \
  -s /data2/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m models/bonsai_local \
  --indoor --eval
```

Train outdoor:

```bash
python train.py \
  -s /data2/peilincai/mesh_datasets/mipnerf360/flowers \
  -m models/flowers_local \
  --outdoor --eval
```

Render and evaluate:

```bash
python render.py --iteration 30000 \
  -s /data2/peilincai/mesh_datasets/mipnerf360/flowers \
  -m models/flowers_local --eval --skip_train

python metrics.py -m models/flowers_local
```
