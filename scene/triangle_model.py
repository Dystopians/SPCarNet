#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import numpy as np
from collections import defaultdict, deque
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
import math
from simple_knn._C import distCUDA2
import math
import rdel



def random_rotation_matrices(num_matrices, device='cpu'):
    """
    Returns a tensor of shape (num_matrices, 3, 3) containing 
    random 3D rotation matrices.
    """
    axis = torch.randn(num_matrices, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    
    angles = 2.0 * math.pi * torch.rand(num_matrices, device=device)
    sin_t = torch.sin(angles)
    cos_t = torch.cos(angles)
    
    K = torch.zeros(num_matrices, 3, 3, device=device)
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1] = -uz
    K[:, 0, 2] =  uy
    K[:, 1, 0] =  uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] =  ux
    
    K2 = K.bmm(K)
    
    I = torch.eye(3, device=device).unsqueeze(0).expand(num_matrices, -1, -1)
    
    sin_term = sin_t.view(-1, 1, 1) * K
    cos_term = (1.0 - cos_t).view(-1, 1, 1) * K2
    
    return I + sin_term + cos_term


def fibonacci_directions(nb_points, device='cpu'):
    """
    Generate nb_points points on the unit sphere using a Fibonacci approach.
    Returns a tensor of shape (nb_points, 3).
    """
    directions = []
    for i in range(nb_points):
        z_coord = 1.0 - (2.0 * i / (nb_points - 1))
        z_coord = torch.tensor(z_coord, device=device)
        radius_xy = torch.sqrt(1.0 - z_coord * z_coord)
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        
        x_unit = radius_xy * torch.cos(torch.tensor(theta, device=device))
        y_unit = radius_xy * torch.sin(torch.tensor(theta, device=device))
        
        directions.append(torch.stack([x_unit, y_unit, z_coord]))
    return torch.stack(directions, dim=0)


def generate_triangles_in_chunks(x, y, z, radii, nb_points=3, chunk_size=2000):
    device = x.device

    num_centers = x.shape[0]

    base_dirs = fibonacci_directions(nb_points, device=device)
    out_points = torch.zeros(num_centers, nb_points, 3, device=device)

    for start_idx in range(0, num_centers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_centers)

        x_chunk = x[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        z_chunk = z[start_idx:end_idx]
        r_chunk = radii[start_idx:end_idx]

        chunk_size_actual = x_chunk.shape[0]

        R_chunk = random_rotation_matrices(chunk_size_actual, device=device)

        for i in range(nb_points):
            dir_i = base_dirs[i]

            dir_i_expanded = dir_i.view(1, 3, 1).expand(chunk_size_actual, -1, -1)

            rotated = R_chunk.bmm(dir_i_expanded)
            rotated = rotated.squeeze(-1)

            scaled = rotated * r_chunk.view(-1, 1)

            centers = torch.stack([x_chunk, y_chunk, z_chunk], dim=1)

            result_pts = centers + scaled

            out_points[start_idx:end_idx, i, :] = result_pts

    return out_points


class TriangleModel:

    def setup_functions(self):
        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int, use_sparse_adam : bool = False):

        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = 0
        self.pixel_count = 0
        self.importance_score = 0
        self.add_percentage = 1.0

        self.scaling = 1
        self._temporary_active_mask = None

        self.laplacian_update_freq = 50  # Update every 50 iterations
        self.iteration_count = 0

        self.use_sparse_adam = use_sparse_adam

        self.setup_functions()

    def get_temporary_active_mask(self):
        return self._temporary_active_mask

    def set_temporary_active_mask(self, active_mask: torch.Tensor):
        mask = active_mask.to(device=self._triangle_indices.device, dtype=torch.bool).contiguous()
        if int(mask.numel()) != int(self._triangle_indices.shape[0]):
            raise ValueError(
                "temporary active mask size mismatch: got {}, expected {}".format(
                    int(mask.numel()), int(self._triangle_indices.shape[0])
                )
            )
        self._temporary_active_mask = mask

    def clear_temporary_active_mask(self):
        self._temporary_active_mask = None

    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size
        point_cloud_state_dict["pixel_count"] = self.pixel_count

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))


    def load_ply_file(self, path, device="cuda", active_sh_degree=3, assume_yup_to_zup=False, training_args=None):
        import trimesh
        """
        Load vertices, faces, and SH features from a PLY file into the current object.
        Fields not derivable from the PLY, like vertex_weight, sigma, importance_score, are ignored.
        """
        SH_C0 = 0.28209479177387814

        def _to_float01(colors_np):
            if colors_np is None:
                return None
            if colors_np.ndim != 2 or colors_np.shape[1] < 3:
                return None
            rgb = colors_np[:, :3]
            if rgb.dtype == np.uint8:
                return rgb.astype(np.float32) / 255.0
            return rgb.astype(np.float32)

        ply_path = path if path.lower().endswith(".ply") else os.path.join(path, "mesh.ply")
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY not found at '{ply_path}'")

        mesh = trimesh.load(ply_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # merge scene geometry into one mesh if needed
            try:
                mesh = trimesh.util.concatenate([g for g in mesh.dump()])
            except Exception as e:
                raise ValueError("Loaded PLY is not a Trimesh and could not be merged") from e

        verts_np = mesh.vertices.astype(np.float32).copy()
        if assume_yup_to_zup:
            # If your PLY is Y-up and you want Z-up internally, apply inverse of (x, y, z)->(x, z, -y)
            y = verts_np[:, 1].copy()
            z = verts_np[:, 2].copy()
            verts_np[:, 1] = -z
            verts_np[:, 2] = y

        faces_np = mesh.faces.astype(np.int32).copy() if mesh.faces is not None else np.empty((0, 3), np.int32)

        colors01 = None
        if getattr(mesh, "visual", None) is not None and hasattr(mesh.visual, "vertex_colors"):
            colors01 = _to_float01(mesh.visual.vertex_colors)

        V = int(verts_np.shape[0])
        verts = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
        faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)

        # features_dc: infer from colors if available, otherwise default to gray which maps to f_dc=0
        if colors01 is not None and colors01.shape[0] == V:
            f_dc_rgb = ((colors01 - 0.5) / SH_C0).clip(-4.0, 4.0).astype(np.float32)
        else:
            f_dc_rgb = np.zeros((V, 3), dtype=np.float32)
        features_dc = torch.from_numpy(f_dc_rgb).to(device=device, dtype=torch.float32).unsqueeze(1).detach().clone().requires_grad_(True)

        # features_rest: zeros with shape [V, (deg+1)^2 - 1, 3]
        deg = int(active_sh_degree)
        num_coeff_total = (deg + 1) ** 2
        num_rest = max(0, num_coeff_total - 1)
        features_rest = torch.zeros((V, num_rest, 3), device=device, dtype=torch.float32, requires_grad=True)

        # Assign to object
        self.vertices = verts.requires_grad_(True)
        self._triangle_indices = faces
        self.active_sh_degree = deg
        self._features_dc = features_dc.requires_grad_(True)
        self._features_rest = features_rest.requires_grad_(True)

        opacity_weight = 1.0
        self.opacity_floor = 0.9999
        vert_weight = inverse_sigmoid(opacity_weight * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))
        self._sigma = self.inverse_exponential_activation(0.0001)

        # Optional, quick report
        print(f"Loaded PLY: {ply_path}")
        print(f"Vertices: {V}, Faces: {faces.shape[0]}, SH degree: {deg}, features_rest per color: {num_rest}")

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

        if training_args != None:
            self.optimizer = None
            self.triangle_scheduler_args = None
            param_groups = [
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self.vertices], 'lr': training_args.lr_triangles_points_init, "name": "vertices"},
                {'params': [self.vertex_weight], 'lr': training_args.weight_lr, "name": "vertex_weight"}
            ]
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

            self.triangle_scheduler_args = get_expon_lr_func(lr_init=training_args.lr_triangles_points_init,
                                                            lr_final=training_args.lr_triangles_points_init/100,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)



    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.75):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = state["active_sh_degree"]
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest      = state["features_rest"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        
        print("triangles: ", self._triangle_indices.shape)
        print("vertices: ", self.vertices.shape)

        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################

        self.opacity_floor = 0.999
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0016, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0016 / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': 0.0001, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) # torch.optim.SGD(param_groups, lr=0.0, momentum=0.0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property 
    def get_triangles_points(self): 
        return self._triangles

    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_vertices(self):
        return self.vertices

    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        # main features
        features_dc   = self._features_dc
        features_rest = self._features_rest
        feats_main = torch.cat((features_dc, features_rest), dim=1)  # [Vmain, F, 3]
        return feats_main
       
    @property
    def get_vertex_weight(self):
        main_w = self.opacity_activation(self.vertex_weight)
        return main_w

    def oneupSHdegree(self):
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, opacity : float, set_sigma : float):

        init_size = 2.23
        nb_points = 3  # 3 verts per triangle

        # --- Load PCD ---
        pcd_points = np.asarray(pcd.points)            # [N,3] (CPU, np)
        pcd_colors = np.asarray(pcd.colors)            # [N,3] (CPU, np)

        fused_point_cloud = torch.tensor(pcd_points, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_rgb   = torch.tensor(pcd_colors, dtype=torch.float32, device="cuda")  # [N,3]
        fused_color_sh    = RGB2SH(fused_color_rgb)                                       # [N,3]

        # SH features per *vertex* will be built after expansion to 3N
        # but we keep your original features layout
        base_feat_dim = (self.max_sh_degree + 1) ** 2

        # --- Scene size (same logic) ---
        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2]
        width  = x.max() - x.min()
        height = y.max() - y.min()
        depth  = z.max() - z.min()
        scene_size = torch.max(torch.stack([width, height, depth]))
        if scene_size.item() > 300:
            print("Scene is large, we increase the threshold")
            self.large = True

        # --- Per-point radii using your GPU NN distance (distCUDA2 returns squared NN dist) ---
        total_points = pcd_points  # naming for clarity
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()),
            1e-7
        )  # [N]
        radii = init_size * torch.sqrt(dist2).unsqueeze(1)  # [N,1]

        # --- Create 1 independent triangle per point (returns [N,3,3]) ---
        points_per_triangle = generate_triangles_in_chunks(x, y, z, radii, nb_points=nb_points)  # [N,3,3]

        # --- Flatten to vertex buffer and build triangle indices ---
        N = fused_point_cloud.shape[0]   # number of triangles
        _points = points_per_triangle.reshape(N * nb_points, 3).contiguous()  # [3N,3]
        faces = torch.arange(N * nb_points, device=_points.device, dtype=torch.int64).view(N, nb_points)  # [N,3]
        faces = faces.to(torch.int32)  # match your Delaunay path dtype

        # --- Per-vertex SH features (color from source point, repeated to its 3 verts) ---
        per_vertex_color_sh = fused_color_sh.repeat_interleave(nb_points, dim=0)  # [3N,3]
        features = torch.zeros((per_vertex_color_sh.shape[0], 3, base_feat_dim),
                            dtype=torch.float32, device="cuda")                 # [3N,3,F]
        features[:, :3, 0] = per_vertex_color_sh
        # features[:, 3:, 1:] stays zero (no higher SH bands initialized)

        # --- Parameters aligned with your Delaunay initializer ---
        self.vertices = nn.Parameter(_points.requires_grad_(True))                 # [3N,3]
        self._triangle_indices = faces                                            # [N,3] int32

        vert_weight = inverse_sigmoid(
            opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float32, device="cuda")
        )
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))        # [3N,1]

        # solid triangles
        self._sigma = self.inverse_exponential_activation(set_sigma)

        # SH feature tensors (transpose to [3N, 1, F] and [3N, (3-1)=2, F] like your code)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        # Per-triangle buffers (match Delaunay sizing by triangles count)
        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float32, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")

      
  
    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):
      
        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    
    def update_learning_rate_delaunay(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = self.triangle_scheduler_args(iteration)
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles):
        # Create dictionary of new tensors to append
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        
        # Append new tensors to optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # Update model parameters
        self.vertices = optimizable_tensors["vertices"]
        self.vertex_weight = optimizable_tensors["vertex_weight"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        # Update triangle indices
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")



    def _update_params_fast(self, selected_indices, iteration):
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_ab, m_bc])
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0
        
        opacity_u = self.opacity_activation(self.vertex_weight[u])
        opacity_v = self.opacity_activation(self.vertex_weight[v])
        avg_opacity = (opacity_u + opacity_v) / 2.0
        avg_opacity = torch.clamp(avg_opacity, self.opacity_floor + self.eps, 1 - self.eps)
        new_vertex_weight = self.inverse_opacity_activation(avg_opacity)

        new_triangles = subdivided_triangles
        
        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # Prune optimizer state
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor


    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]
            if isinstance(self.pixel_count, torch.Tensor) and self.pixel_count.numel() > 0:
                self.pixel_count = self.pixel_count[valid_tris]
            

        # Prune vertex-related parameters using the initial mask
        self._prune_vertex_optimizer(vertex_mask)

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._prune_vertex_optimizer(mask_referenced)

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()
        self.clear_temporary_active_mask()

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask]
        self._triangle_indices = self._triangle_indices.to(torch.int32)
        self.image_size = self.image_size[mask]
        self.importance_score = self.importance_score[mask]
        self.pixel_count = self.pixel_count[mask]
        self.clear_temporary_active_mask()
        

    def _sample_alives(self, probs, num, alive_indices=None):
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        return sampled_idxs        

    def add_new_gs(self, iteration, cap_max, splitt_large_triangles):

        current_num_points = self.vertices.shape[0]
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Find indexes based on proba
        triangle_transp = self.importance_score
        probs = triangle_transp.squeeze()

        areas = self.triangle_areas().squeeze()
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(self.image_size < self.size_probs_zero_image_space, torch.zeros_like(probs), probs) # dont splitt if smaller than 10

        rand_idx = self._sample_alives(probs=probs, num=num_gs)

        # Split the largest triangles
        split_large = splitt_large_triangles
        k = min(split_large, areas.numel())  
        _, top_idx = torch.topk(areas, k, largest=True, sorted=False)

        # 3) combine and deduplicate
        add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)

        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles) = self._update_params_fast(add_idx, iteration)

        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles)

        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        mask[add_idx] = False
        self.prune_triangles(mask)



    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True):
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            mask = self.vertices.shape[0]
            y = self.get_vertex_weight[:mask].detach()
            y = y.clamp(new_m + self.eps, 1.0 - self.eps)   # clamp to the *new* floor
        self.opacity_floor = new_m
        new_logits = self.inverse_opacity_activation(y)
        with torch.no_grad():
            self.vertex_weight.data.copy_(new_logits)


    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas

    def _rebuild_optimizer_after_topology_change(self):
        """
        Rebuild optimizer after topology changes.
        Momentum buffers are reset intentionally because the parameter shapes changed.
        """
        if self.optimizer is None:
            return

        lr_by_name = {g.get("name", f"group_{i}"): g.get("lr", 0.0) for i, g in enumerate(self.optimizer.param_groups)}
        param_groups = [
            {'params': [self._features_dc],   'lr': lr_by_name.get("f_dc", 0.0016), "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_by_name.get("f_rest", 0.0016 / 20.0), "name": "f_rest"},
            {'params': [self.vertices],       'lr': lr_by_name.get("vertices", 0.0001), "name": "vertices"},
            {'params': [self.vertex_weight],  'lr': lr_by_name.get("vertex_weight", 0.0), "name": "vertex_weight"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def optimize_ground_planar_patches(
        self,
        up_axis="auto",
        max_ground_tilt_deg=30.0,
        max_neighbor_normal_deg=20.0,
        max_neighbor_height_delta=0.10,
        min_region_triangles=80,
        min_region_area=0.05,
        max_plane_residual=0.03,
        residual_quantile=0.95,
        snap_cell_size=0.05,
        near_center=None,
        near_radius=-1.0,
        enable_global_snap=False,
        global_height_bin=0.05,
        allow_boundary_snap=False,
        boundary_snap_max_shift=0.02,
        merge_mode="edge_collapse",
        project_to_plane=True,
        max_project_shift=0.015,
        edge_collapse_length=0.03,
        max_collapse_shift=0.02,
        max_edges_per_region=6000,
        max_candidate_triangles=800000,
        verbose=True,
    ):
        """
        Expression-level planar optimization for large near-planar patches (e.g. ground).

        Strategy:
        1) Select ground-like triangles by normal/up-axis angle.
        2) Build conservative connected regions with neighbor normal gating.
        3) For each valid region, fit a plane and only continue if residual is low.
        4) Keep boundary vertices fixed; only snap interior vertices on-plane and to a 2D grid.
        5) Remap triangles, remove degenerate/duplicate faces, compact vertices.

        This keeps merging strictly inside detected regions and avoids crossing protected boundaries.
        """
        device = self.vertices.device
        verts = self.vertices.detach().cpu().numpy().astype(np.float64)
        tris = self._triangle_indices.detach().cpu().numpy().astype(np.int64)

        if tris.shape[0] == 0 or verts.shape[0] == 0:
            return {"status": "skipped_empty_mesh"}

        # Triangle geometry
        tri_pts = verts[tris]  # [T,3,3]
        ab = tri_pts[:, 1] - tri_pts[:, 0]
        ac = tri_pts[:, 2] - tri_pts[:, 0]
        normals = np.cross(ab, ac)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_n = norm_len[:, 0] > 1e-12
        normals[valid_n] = normals[valid_n] / norm_len[valid_n]
        areas = 0.5 * np.linalg.norm(np.cross(ab, ac), axis=1)

        axis = up_axis.lower()
        axis_map = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
        if axis == "auto":
            # Pick the axis with strongest area-weighted normal concentration.
            axis_scores = {}
            for k, a in axis_map.items():
                axis_scores[k] = float((np.abs(normals @ a) * areas).sum())
            axis = max(axis_scores, key=axis_scores.get)
        elif axis not in axis_map:
            raise ValueError(f"Unsupported up_axis '{up_axis}'. Use one of x/y/z/auto.")
        up = axis_map[axis]

        cos_ground = np.cos(np.deg2rad(max_ground_tilt_deg))
        centroids = tri_pts.mean(axis=1)
        near_mask = np.ones(tris.shape[0], dtype=bool)
        if near_center is not None and near_radius is not None and float(near_radius) > 0:
            c = np.asarray(near_center, dtype=np.float64).reshape(1, 3)
            near_mask = np.linalg.norm(centroids - c, axis=1) <= float(near_radius)
        ground_like = valid_n & (np.abs(normals @ up) >= cos_ground) & near_mask
        candidate_idx = np.where(ground_like)[0]

        if candidate_idx.size == 0:
            return {"status": "skipped_no_ground_candidates", "up_axis_used": axis}

        # Cap candidate size for runtime stability on very large meshes.
        if candidate_idx.size > max_candidate_triangles:
            order = np.argsort(areas[candidate_idx])[::-1]
            candidate_idx = candidate_idx[order[:max_candidate_triangles]]

        cand_normals = normals[candidate_idx]
        cand_tris = tris[candidate_idx]
        cand_areas = areas[candidate_idx]
        cand_centroids = centroids[candidate_idx]

        # Build adjacency by shared edges (only among candidate triangles).
        edge_to_tris = defaultdict(list)
        for t_local, tri in enumerate(cand_tris):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            e0 = (a, b) if a < b else (b, a)
            e1 = (b, c) if b < c else (c, b)
            e2 = (a, c) if a < c else (c, a)
            edge_to_tris[e0].append(t_local)
            edge_to_tris[e1].append(t_local)
            edge_to_tris[e2].append(t_local)

        neigh_cos = np.cos(np.deg2rad(max_neighbor_normal_deg))
        neighbors = [[] for _ in range(cand_tris.shape[0])]
        for tri_list in edge_to_tris.values():
            if len(tri_list) != 2:
                continue
            i, j = tri_list
            normal_ok = np.abs(np.dot(cand_normals[i], cand_normals[j])) >= neigh_cos
            height_ok = np.abs(np.dot((cand_centroids[i] - cand_centroids[j]), up)) <= float(max_neighbor_height_delta)
            if normal_ok and height_ok:
                neighbors[i].append(j)
                neighbors[j].append(i)

        # Region growing
        visited = np.zeros(cand_tris.shape[0], dtype=bool)
        regions = []
        for seed in range(cand_tris.shape[0]):
            if visited[seed]:
                continue
            q = deque([seed])
            visited[seed] = True
            region = []
            while q:
                cur = q.popleft()
                region.append(cur)
                for nb in neighbors[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            regions.append(region)

        remap = np.arange(verts.shape[0], dtype=np.int64)
        parent = np.arange(verts.shape[0], dtype=np.int64)
        merged_regions = 0
        rejected_small = 0
        rejected_area = 0
        rejected_plane_tilt = 0
        rejected_residual = 0
        edge_collapses = 0

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Global tangent frame for optional cross-region snapping
        if np.abs(up[2]) < 0.99:
            g0 = np.cross(up, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        else:
            g0 = np.cross(up, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        g0 = g0 / (np.linalg.norm(g0) + 1e-12)
        g1 = np.cross(up, g0)
        g1 = g1 / (np.linalg.norm(g1) + 1e-12)
        global_origin = centroids[candidate_idx].mean(axis=0) if candidate_idx.size > 0 else np.zeros(3, dtype=np.float64)
        global_cell_to_rep = {}

        for region in regions:
            if len(region) < min_region_triangles:
                rejected_small += 1
                continue

            region = np.asarray(region, dtype=np.int64)
            region_area = float(cand_areas[region].sum())
            if region_area < float(min_region_area):
                rejected_area += 1
                continue

            region_tris = cand_tris[region]  # [R,3]
            region_vert_ids = np.unique(region_tris.reshape(-1))
            region_points = verts[region_vert_ids]
            if region_points.shape[0] < 8:
                continue

            # Plane fit
            centroid = region_points.mean(axis=0)
            centered = region_points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            plane_n = vh[-1]
            if np.abs(np.dot(plane_n, up)) < cos_ground:
                rejected_plane_tilt += 1
                continue

            # Keep orientation stable
            if np.dot(plane_n, up) < 0:
                plane_n = -plane_n

            dists = np.abs(centered @ plane_n)
            if np.quantile(dists, float(residual_quantile)) > max_plane_residual:
                rejected_residual += 1
                continue

            # Region boundary vertices: edges appearing once in this region.
            e01 = np.sort(region_tris[:, [0, 1]], axis=1)
            e12 = np.sort(region_tris[:, [1, 2]], axis=1)
            e02 = np.sort(region_tris[:, [0, 2]], axis=1)
            region_edges = np.concatenate([e01, e12, e02], axis=0)
            uniq_edges, counts = np.unique(region_edges, axis=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
            boundary_vertices = set(boundary_edges.reshape(-1).tolist())

            interior_vertices = [vid for vid in region_vert_ids.tolist() if vid not in boundary_vertices]
            candidate_vertices = interior_vertices

            if allow_boundary_snap:
                # Conservative boundary snapping: only very small on-plane displacement allowed.
                extra = []
                for vid in boundary_vertices:
                    p = verts[vid]
                    p_proj = p - np.dot((p - centroid), plane_n) * plane_n
                    shift = np.linalg.norm(p_proj - p)
                    if shift <= float(boundary_snap_max_shift):
                        extra.append(int(vid))
                candidate_vertices = interior_vertices + extra

            if len(candidate_vertices) < 4:
                continue

            # Build in-plane basis
            t0 = vh[0]
            t1 = vh[1]
            if np.linalg.norm(t0) < 1e-12 or np.linalg.norm(t1) < 1e-12:
                continue

            if merge_mode == "snap":
                # Conservative in-plane snapping.
                cell_to_rep = {}
                for vid in candidate_vertices:
                    p = verts[vid]

                    if enable_global_snap:
                        relg = p - global_origin
                        gu = np.dot(relg, g0)
                        gv = np.dot(relg, g1)
                        gh = np.dot(relg, up)
                        key = (
                            int(np.floor(gu / snap_cell_size)),
                            int(np.floor(gv / snap_cell_size)),
                            int(np.floor(gh / max(float(global_height_bin), 1e-6))),
                        )
                        if key not in global_cell_to_rep:
                            global_cell_to_rep[key] = vid
                        rep = global_cell_to_rep[key]
                    else:
                        rel = p - centroid
                        u = np.dot(rel, t0)
                        v = np.dot(rel, t1)
                        key = (int(np.floor(u / snap_cell_size)), int(np.floor(v / snap_cell_size)))
                        if key not in cell_to_rep:
                            cell_to_rep[key] = vid
                        rep = cell_to_rep[key]

                    remap[vid] = rep
            elif merge_mode == "edge_collapse":
                interior_set = set(interior_vertices)
                active_set = set(candidate_vertices)

                # Step 1: optional local plane projection (small displacement only).
                if project_to_plane:
                    for vid in candidate_vertices:
                        p = verts[vid]
                        p_proj = p - np.dot((p - centroid), plane_n) * plane_n
                        shift = np.linalg.norm(p_proj - p)
                        if shift <= float(max_project_shift):
                            verts[vid] = p_proj

                # Step 2: conservative short-edge collapse inside region.
                e01 = np.sort(region_tris[:, [0, 1]], axis=1)
                e12 = np.sort(region_tris[:, [1, 2]], axis=1)
                e02 = np.sort(region_tris[:, [0, 2]], axis=1)
                reg_edges = np.concatenate([e01, e12, e02], axis=0)
                reg_edges = np.unique(reg_edges, axis=0)

                edge_data = []
                for e in reg_edges:
                    u, v = int(e[0]), int(e[1])
                    if u == v:
                        continue
                    if (u not in active_set) or (v not in active_set):
                        continue
                    if (u not in interior_set) or (v not in interior_set):
                        continue
                    pu = verts[u]
                    pv = verts[v]
                    l = np.linalg.norm(pu - pv)
                    if l > float(edge_collapse_length):
                        continue
                    hdiff = np.abs(np.dot((pu - pv), up))
                    if hdiff > float(max_neighbor_height_delta):
                        continue
                    edge_data.append((l, u, v))

                if edge_data:
                    edge_data.sort(key=lambda x: x[0])
                    if len(edge_data) > int(max_edges_per_region):
                        edge_data = edge_data[: int(max_edges_per_region)]

                    for _, u0, v0 in edge_data:
                        u = _find(u0)
                        v = _find(v0)
                        if u == v:
                            continue
                        pu = verts[u]
                        pv = verts[v]
                        mid = 0.5 * (pu + pv)
                        if np.linalg.norm(mid - pu) > float(max_collapse_shift):
                            continue
                        if np.linalg.norm(mid - pv) > float(max_collapse_shift):
                            continue
                        parent[v] = u
                        verts[u] = mid
                        edge_collapses += 1
            else:
                raise ValueError(f"Unsupported merge_mode '{merge_mode}'. Use 'snap' or 'edge_collapse'.")

            merged_regions += 1

        # Compose collapse map (if edge collapse mode used)
        if merge_mode == "edge_collapse":
            for i in range(parent.shape[0]):
                parent[i] = _find(i)
            remap = parent[remap]

        # Apply vertex remapping
        new_tris = remap[tris]
        deg_mask = (new_tris[:, 0] != new_tris[:, 1]) & (new_tris[:, 1] != new_tris[:, 2]) & (new_tris[:, 0] != new_tris[:, 2])
        new_tris = new_tris[deg_mask]

        # Remove duplicate faces (orientation-agnostic)
        sorted_faces = np.sort(new_tris, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        new_tris = new_tris[np.sort(unique_idx)]

        # Compact vertex arrays
        used = np.unique(new_tris.reshape(-1))
        new_vid = -np.ones(verts.shape[0], dtype=np.int64)
        new_vid[used] = np.arange(used.shape[0], dtype=np.int64)
        compact_tris = new_vid[new_tris].astype(np.int32)

        new_vertices = torch.from_numpy(self.vertices.detach().cpu().numpy()[used]).to(device=device, dtype=torch.float32)
        new_weight = torch.from_numpy(self.vertex_weight.detach().cpu().numpy()[used]).to(device=device, dtype=torch.float32)
        new_fdc = torch.from_numpy(self._features_dc.detach().cpu().numpy()[used]).to(device=device, dtype=torch.float32)
        new_frest = torch.from_numpy(self._features_rest.detach().cpu().numpy()[used]).to(device=device, dtype=torch.float32)

        old_v = int(self.vertices.shape[0])
        old_t = int(self._triangle_indices.shape[0])

        self.vertices = nn.Parameter(new_vertices.requires_grad_(True))
        self.vertex_weight = nn.Parameter(new_weight.requires_grad_(True))
        self._features_dc = nn.Parameter(new_fdc.requires_grad_(True))
        self._features_rest = nn.Parameter(new_frest.requires_grad_(True))
        self._triangle_indices = torch.from_numpy(compact_tris).to(device=device, dtype=torch.int32)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device=device)
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device=device)
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device=device)

        self._rebuild_optimizer_after_topology_change()

        new_v = int(self.vertices.shape[0])
        new_t = int(self._triangle_indices.shape[0])
        stats = {
            "status": "ok",
            "up_axis_used": axis,
            "candidate_triangles": int(candidate_idx.size),
            "regions_total": int(len(regions)),
            "merged_regions": int(merged_regions),
            "rejected_small_region": int(rejected_small),
            "rejected_small_area": int(rejected_area),
            "rejected_plane_tilt": int(rejected_plane_tilt),
            "rejected_plane_residual": int(rejected_residual),
            "enable_global_snap": bool(enable_global_snap),
            "allow_boundary_snap": bool(allow_boundary_snap),
            "merge_mode": merge_mode,
            "edge_collapses": int(edge_collapses),
            "vertices_before": old_v,
            "vertices_after": new_v,
            "triangles_before": old_t,
            "triangles_after": new_t,
            "vertices_reduced": int(old_v - new_v),
            "triangles_reduced": int(old_t - new_t),
        }
        if verbose:
            print("[PlanarMerge] {}".format(stats))
        return stats


    
    def run_restricted_delaunay(self):

        print("Running restricted delaunay... for ", self.vertices.shape[0], " vertices.")

        self._triangle_indices = self._triangle_indices.detach().cpu().numpy()

        faces_ = rdel.run(
            self.vertices.detach().cpu().numpy(),
            self._triangle_indices,
            verbose=False,  # print timings and extra logs if True
            orient=False    # try to consistently orient face normals if True
        )

        self._triangle_indices = torch.as_tensor(np.asarray(faces_, dtype=np.int64), device='cuda').contiguous()
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        print("We have after re-delaunay ", self._triangle_indices.shape[0], " triangles.")

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.pixel_count = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.int, device="cuda")
        self.clear_temporary_active_mask()
