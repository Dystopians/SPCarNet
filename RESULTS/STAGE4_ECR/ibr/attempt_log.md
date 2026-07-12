# IBRNet single-target-view inference attempt log

Date: 2026-07-11T18:30:46-07:00

## 1. Camera convention evidence

Files read:

- `/data/peilincai/IBRNet/ibrnet/data_loaders/llff_data_utils.py`
- `/data/peilincai/IBRNet/ibrnet/data_loaders/llff.py`
- `/data/peilincai/IBRNet/ibrnet/sample_ray.py`
- `/data/peilincai/IBRNet/ibrnet/projection.py`
- `/data/peilincai/IBRNet/ibrnet/render_ray.py`
- `/data/peilincai/IBRNet/ibrnet/render_image.py`
- `/data/peilincai/IBRNet/ibrnet/model.py`
- `/data/peilincai/IBRNet/ibrnet/mlp_network.py`
- `/data/peilincai/IBRNet/config.py`

Decisive lines:

```python
# /data/peilincai/IBRNet/ibrnet/data_loaders/llff_data_utils.py:25-41
def parse_llff_pose(pose):
    '''
    convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    '''
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    c2w_4x4[:, 1:3] *= -1
```

```python
# /data/peilincai/IBRNet/ibrnet/data_loaders/llff_data_utils.py:297-299
# Correct rotation matrix ordering and move variable dim to axis 0
poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
poses = np.moveaxis(poses, -1, 0).astype(np.float32)
```

```python
# /data/peilincai/IBRNet/ibrnet/data_loaders/llff.py:84-86
img_size = rgb.shape[:2]
camera = np.concatenate((list(img_size), intrinsics.flatten(),
                         render_pose.flatten())).astype(np.float32)
```

```python
# /data/peilincai/IBRNet/ibrnet/data_loaders/llff.py:116-119
img_size = src_rgb.shape[:2]
src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                  train_pose.flatten())).astype(np.float32)
src_cameras.append(src_camera)
```

```python
# /data/peilincai/IBRNet/ibrnet/sample_ray.py:27-32
def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w
```

```python
# /data/peilincai/IBRNet/ibrnet/sample_ray.py:87-95
u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
pixels = np.stack((u, v, np.ones_like(u)), axis=0)
rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
```

Conclusion: IBRNet's internal camera arrays store `[H, W, intrinsics(16), c2w(16)]`, and the `c2w` consumed by ray sampling is OpenCV-style. The LLFF loader first corrects LLFF rotation ordering, then `parse_llff_pose` explicitly says it returns "opencv convention" and flips camera columns 1 and 2 there. The job schema already supplies OpenCV axes (x right, y down, z forward), so `/data/peilincai/IBRNet/ibr_infer.py` passes job `c2w` through unchanged and does not apply an additional OpenGL y/z flip.

Determinism evidence:

```python
# /data/peilincai/IBRNet/ibrnet/render_ray.py:40-45
if det:
    u = torch.linspace(0., 1., N_samples, device=bins.device)
    u = u.unsqueeze(0).repeat(bins.shape[0], 1)
else:
    u = torch.rand(bins.shape[0], N_samples, device=bins.device)
```

```python
# /data/peilincai/IBRNet/ibrnet/render_ray.py:104-111
if not det:
    ...
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand
```

`ibr_infer.py` calls `render_single_image(..., det=True)` and uses `model.switch_to_eval()` plus `torch.no_grad()`, so the eval render path does not use random depth/PDF sampling.

## 2. Compatibility edits made

New deliverable:

- `/data/peilincai/IBRNet/ibr_infer.py`: added a standalone CLI with `python ibr_infer.py --job <job.json> [--device cuda|cpu]`, checkpoint/model construction through `IBRNetModel`, source image loading without resizing, source camera arrays in `[H, W, intrinsics(16), c2w(16)]`, OpenCV c2w pass-through comment, adaptive/configurable chunking, coarse+fine rendering, fine RGB PNG output, and per-image timing.

Compatibility edits:

- `/data/peilincai/IBRNet/ibrnet/model.py:31-54`: before, model construction hard-coded `cuda:<local_rank>` and `feature_net.cuda()`; after, `IBRNetModel(..., device=None)` resolves CPU/CUDA and sends coarse, fine, and feature networks to that device. Reason: CPU smoke test cannot construct CUDA modules.
- `/data/peilincai/IBRNet/ibrnet/model.py:83-101`: before, distributed wrappers always used CUDA `device_ids` and `output_device`; after, those are only set for CUDA devices. Reason: keep the constructor valid when the selected device is CPU.
- `/data/peilincai/IBRNet/ibrnet/model.py:127-132`: before, checkpoint loading used `torch.load(filename)` or CUDA-only `map_location`; after, it uses the selected `map_location` and `torch.load(..., weights_only=False)`. Reason: trusted local torch-1.x checkpoint dictionaries must load under PyTorch 2.x, including optimizer/scheduler keys if requested.
- `/data/peilincai/IBRNet/ibrnet/sample_ray.py:51`: before, the sampler stored the raw device argument; after, it stores `torch.device(device)`. Reason: normalize device handling for CPU and CUDA.
- `/data/peilincai/IBRNet/ibrnet/sample_ray.py:91`: before, pixel coordinates were always created as CPU tensors; after, they are moved to `c2w.device`. Reason: avoid CPU/GPU tensor mismatch when the sampler is later used on CUDA.
- `/data/peilincai/IBRNet/ibrnet/sample_ray.py:99-107`: before, `get_all()` hard-coded `.cuda()` on rays, cameras, depth range, RGBs, and source cameras; after, it uses `.to(self.device)`. Reason: make full-image inference work on CPU.
- `/data/peilincai/IBRNet/ibrnet/sample_ray.py:148-155`: before, `random_sample()` hard-coded `.cuda()` for training ray batches; after, it uses `.to(self.device)`. Reason: keep the shared sampler device-parametric beyond this CLI path.
- `/data/peilincai/IBRNet/ibrnet/mlp_network.py:200`: before, positional encoding was a plain tensor attribute; after, it is a non-persistent buffer. Reason: it follows module device placement without adding checkpoint state keys.
- `/data/peilincai/IBRNet/ibrnet/mlp_network.py:216-218`: before, positional encoding was built on `cuda:<local_rank>`; after, it uses `args.device` when available, otherwise CUDA if available, otherwise CPU. Reason: CPU model construction otherwise failed before checkpoint loading.
- `/data/peilincai/IBRNet/ibrnet/projection.py:37-40`: before, `torch.tensor([w-1., h-1.])` constructed a tensor from tensors; after, `torch.stack((w - 1., h - 1.))` preserves tensor semantics cleanly. Reason: avoid modern PyTorch tensor-construction pitfalls and keep device placement explicit.

No dependency installation was needed. The required venv already had Python 3.11.14, torch 2.7.1+cu126, numpy 2.4.2, imageio 2.37.3, configargparse, and Pillow available.

## 3. CPU smoke test result

Synthetic job generated under `/tmp/ibr_smoke/`:

- `/tmp/ibr_smoke/src_0.png`
- `/tmp/ibr_smoke/src_1.png`
- `/tmp/ibr_smoke/job.json`
- target size: width 64, height 48
- depth range: `[2.0, 6.0]`
- output path: `/tmp/ibr_smoke/out.png`

Required help check:

```bash
/data/peilincai/gems_stage1/ibr_cell/venv/bin/python ibr_infer.py --help
```

Result: exit code 0, printed usage text with `--job`, `--device`, `--checkpoint`, `--chunk-size`, `--n-samples`, and `--n-importance`.

Syntax/import check:

```bash
/data/peilincai/gems_stage1/ibr_cell/venv/bin/python -m py_compile ibr_infer.py ibrnet/model.py ibrnet/sample_ray.py ibrnet/mlp_network.py ibrnet/projection.py
```

Result: exit code 0.

End-to-end CPU command:

```bash
/usr/bin/time -f 'wall_clock_seconds %e' /data/peilincai/gems_stage1/ibr_cell/venv/bin/python ibr_infer.py --job /tmp/ibr_smoke/job.json --device cpu
```

Output:

```text
Reloading from /data/peilincai/IBRNet/pretrained/model_255000.pth, starting at step=255000
rendered /tmp/ibr_smoke/out.png (64x48) in 52.893s [load=0.023s, features=3.725s, render=47.218s, save=0.034s, chunk_size=256, device=cpu]
wall_clock_seconds 55.03
```

PNG inspection:

```text
path /tmp/ibr_smoke/out.png
shape (48, 64, 3)
dtype uint8
min 2 max 250 mean 123.88172743055556
matches_target True
```

Result: PASS. The output PNG exists, is nonblank, and has actual dimensions `(48, 64, 3)`, matching target width 64 and height 48.

## 4. Remaining risk flags for later GPU use

- CUDA/GPU execution was not verified because this sandbox has no CUDA devices (`torch.cuda.is_available()` was false).
- The adaptive render path retries on Python `RuntimeError` messages containing "out of memory"; a hard process kill from host memory pressure cannot be caught.
- Default automatic chunk sizes are conservative guesses: 256 rays on CPU and 1024 rays on CUDA. Real scenes with many source views or high resolution may need `--chunk-size` tuning.
- Source images are loaded at native resolution and never resized. If source dimensions differ, the CLI center-pads to a common tensor size and shifts principal points accordingly; this compatibility path passed code review but was not covered by the equal-size synthetic smoke job.
- No network fetches or package installs were attempted. No missing dependencies surfaced during the verified CPU path.
