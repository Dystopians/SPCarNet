#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import cv2
import os
from utils.ground_mask_utils import (
    resolve_ground_mask_path,
    load_ground_mask_binary,
    resize_ground_mask,
    maybe_save_mask_overlay,
)

WARNED = False
_MISSING_GROUND_MASK_WARNED = set()

import numpy as np, torch
from pathlib import Path

def to_depth_tensor(depth_in):
    if depth_in is None:
        return None

    # If it's a path, load the .npy
    if isinstance(depth_in, (str, Path)):
        arr = np.load(depth_in).astype(np.float32)
        if arr.ndim == 2: arr = arr[..., None]         # [H,W] -> [H,W,1]
        t = torch.from_numpy(arr)                      # [H,W,1]
        return t.permute(2,0,1).contiguous()           # [1,H,W]

    # If it's already a numpy array
    if isinstance(depth_in, np.ndarray):
        arr = depth_in.astype(np.float32)
        if arr.ndim == 2: arr = arr[..., None]
        t = torch.from_numpy(arr)
        return t.permute(2,0,1).contiguous()

    # If it's already a tensor
    if torch.is_tensor(depth_in):
        t = depth_in.float()
        # allow [H,W], [H,W,1], [1,H,W]
        if t.ndim == 2:    t = t.unsqueeze(0)          # [H,W]   -> [1,H,W]
        elif t.ndim == 3 and t.shape[-1] == 1: t = t.permute(2,0,1)  # [H,W,1] -> [1,H,W]
        # if it's already [1,H,W], keep it
        return t.contiguous()

    raise TypeError(f"Unsupported depth type: {type(depth_in)}")

def loadCam(args, id, cam_info, resolution_scale):


    if cam_info.depth_path != "":
        try:
            invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    normal_map = getattr(cam_info, 'normal_map', None)
    if normal_map is not None:
        normal_map = torch.from_numpy(normal_map).permute(2, 0, 1).float()  # [3, H, W]
    else:
        normal_map = None

    ground_mask = None
    use_ground_masks = bool(getattr(args, "ground_masks", False) or getattr(args, "enable_ground_masks", False)) and bool(getattr(args, "ground_mask_dir", ""))
    if use_ground_masks:
        mask_path = resolve_ground_mask_path(
            source_path=getattr(args, "source_path", ""),
            mask_dir=getattr(args, "ground_mask_dir", ""),
            image_name=cam_info.image_name,
            image_path=cam_info.image_path,
            matching=getattr(args, "ground_mask_matching", "auto"),
            suffix=getattr(args, "ground_mask_suffix", ".png"),
            missing_strategy=getattr(args, "ground_mask_missing_strategy", "empty"),
            nearest_max_gap=getattr(args, "ground_mask_nearest_max_gap", 6),
        )
        if mask_path is None:
            image_key = cam_info.image_name
            if image_key not in _MISSING_GROUND_MASK_WARNED:
                print(f"[GroundMask] Missing mask for image '{cam_info.image_name}'. Using empty mask fallback.")
                _MISSING_GROUND_MASK_WARNED.add(image_key)
            ground_mask = torch.zeros((resolution[1], resolution[0]), dtype=torch.bool)
        else:
            try:
                ground_mask_raw = load_ground_mask_binary(
                    mask_path=mask_path,
                    threshold=getattr(args, "ground_mask_threshold", 127),
                    label_value=getattr(args, "ground_mask_label_value", -1),
                    label_rgb=getattr(args, "ground_mask_label_rgb", ""),
                )
                ground_mask = resize_ground_mask(ground_mask_raw, out_h=resolution[1], out_w=resolution[0])
            except Exception as exc:
                image_key = cam_info.image_name
                if image_key not in _MISSING_GROUND_MASK_WARNED:
                    print(
                        f"[GroundMask] Failed to load mask '{mask_path}' for image '{cam_info.image_name}' "
                        f"({exc}). Using empty mask fallback."
                    )
                    _MISSING_GROUND_MASK_WARNED.add(image_key)
                ground_mask = torch.zeros((resolution[1], resolution[0]), dtype=torch.bool)

        assert ground_mask.shape[0] == gt_image.shape[1] and ground_mask.shape[1] == gt_image.shape[2], (
            f"Ground mask shape {tuple(ground_mask.shape)} does not match image shape "
            f"{tuple(gt_image.shape[1:])} for '{cam_info.image_name}'."
        )

        debug_enabled = bool(getattr(args, "ground_mask_debug_vis", False))
        debug_max = int(getattr(args, "ground_mask_debug_max", 8))
        debug_dir = getattr(args, "ground_mask_debug_dir", "")
        if not debug_dir:
            debug_dir = os.path.join(getattr(args, "model_path", "."), "ground_mask_debug")
        maybe_save_mask_overlay(
            rgb_chw=gt_image,
            mask_hw=ground_mask,
            image_name=cam_info.image_name,
            debug_enabled=debug_enabled,
            debug_dir=debug_dir,
            max_examples=debug_max,
        )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,  depth_params=cam_info.depth_params, invdepthmap=invdepthmap,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, normal_map=normal_map, ground_mask=ground_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry