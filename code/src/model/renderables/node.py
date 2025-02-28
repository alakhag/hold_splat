import torch.nn as nn

import src.engine.volsdf_utils as volsdf_utils
from src.engine.rendering import render_color

from ...engine.density import LaplaceDensity
from ...engine.ray_sampler import ErrorBoundSampler
from ...networks.shape_net import ImplicitNet
from ...networks.texture_net import RenderingNet

from gaussian_renderer import render
from typing import NamedTuple
import trimesh

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


class Node(nn.Module):
    def __init__(
        self,
        args,
        opt,
        specs,
        sdf_bounding_sphere,
        implicit_network_opt,
        rendering_network_opt,
        deformer,
        server,
        class_id,
        node_id,
        params,
    ):
        super(Node, self).__init__()
        self.args = args
        self.specs = specs
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.implicit_network = ImplicitNet(implicit_network_opt, args, specs)
        self.rendering_network = RenderingNet(rendering_network_opt, args, specs)
        self.ray_sampler = ErrorBoundSampler(
            self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler
        )
        self.density = LaplaceDensity(**opt.density)
        self.deformer = deformer
        self.server = server
        self.class_id = class_id
        self.node_id = node_id
        self.params = params

    def meshing_cano(self, pose=None):
        return None

    def sample_points(self, input):
        raise NotImplementedError("Derived classes should implement this method.")

    def forward(self, input):
        if "time_code" in input:
            time_code = input["time_code"]
        else:
            time_code = None
        sample_dict = self.sample_points(input)

        # compute canonical SDF and features
        (
            sdf_output,
            canonical_points,
            feature_vectors,
        ) = volsdf_utils.sdf_func_with_deformer(
            self.deformer,
            self.implicit_network,
            self.training,
            sample_dict["points"].reshape(-1, 3),
            sample_dict["deform_info"],
        )
        num_samples = sample_dict["z_vals"].shape[1]
        color, normal, semantics = self.render(
            sample_dict, num_samples, canonical_points, feature_vectors, time_code
        )
        self.device = color.device

        num_samples = color.shape[1]
        density = self.density(sdf_output).view(-1, num_samples, 1)
        sample_dict["canonical_pts"] = canonical_points.view(
            sample_dict["batch_size"], sample_dict["num_pixels"], num_samples, 3
        )
        # color, normal, density, semantics
        factors = {
            "color": color,
            "normal": normal,
            "density": density,
            "semantics": semantics,
            "z_vals": sample_dict["z_vals"],
        }
        return factors, sample_dict

    def render(
        self, sample_dict, num_samples, canonical_points, feature_vectors, time_code
    ):
        color, normal, semantics = render_color(
            self.deformer,
            self.implicit_network,
            self.rendering_network,
            sample_dict["ray_dirs"],
            sample_dict["cond"],
            sample_dict["tfs"],
            canonical_points,
            feature_vectors,
            self.training,
            num_samples,
            self.class_id,
            time_code,
        )
        return color, normal, semantics

    def step_embedding(self):
        self.implicit_network.embedder_obj.step()

class Splats:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree=3,
        optimizer_type="default",
        server=None,
        deformer=None,
        node_id=None,
        class_id=0
    ):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.deformer = deformer
        self.server = server
        self.node_id = node_id
        self.class_id = class_id

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_normal(self):
        """
        Compute the surface normals from the Gaussian's rotation.
        Assumes that build_rotation(self._rotation) returns a (N, 3, 3) rotation matrix.
        """
        # Define the canonical normal (e.g., pointing up in canonical space)
        canonical_normal = torch.tensor([0.0, 0.0, 1.0], device=self._rotation.device)
        canonical_normal = canonical_normal.expand(self._rotation.shape[0], 3)  # (N, 3)
        
        # Get rotation matrices for each Gaussian
        R = build_rotation(self._rotation)  # Expected shape: (N, 3, 3)
        
        # Compute normals by applying the rotation to the canonical normal
        normals = torch.bmm(R, canonical_normal.unsqueeze(-1)).squeeze(-1)

        # normals = canonical_normal
        
        # Ensure unit length (optional but recommended)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        
        return normals

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normal().detach().cpu().numpy() 
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_pcd(self):
        print (f"Generating random points for {self.node_id}")
        num_pts = 30_000
        

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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # def _rotation_matrix_from_vectors(self, a, b):
    #     """
    #     Compute the rotation matrix (3x3) that rotates vector a to vector b.
    #     a and b are torch tensors of shape (3,).
    #     """
    #     a = a / torch.norm(a) # (3,)
    #     b = b / torch.norm(b) # (3,)
    #     v = torch.cross(a, b) # (3,)
    #     s = torch.norm(v) # scalar
    #     c = torch.dot(a, b) # scalar
    #     if s < 1e-8:
    #         return torch.eye(3, device=a.device)
    #     vx = torch.tensor([[    0, -v[2],  v[1]],
    #                        [ v[2],     0, -v[0]],
    #                        [-v[1],  v[0],     0]], device=a.device)
    #     R = torch.eye(3, device=a.device) + vx + vx @ vx * ((1 - c) / (s**2))
    #     return R

    def _rotation_matrix_from_vectors(self, a, b):
        """
        a is (N,3) vector
        b is (3,) vector
        output is (N,3,3) matrix
        """
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        a = a / torch.norm(a, dim=1, keepdim=True) # (N,3)
        b = b / torch.norm(b) # (3,)
        b = b.unsqueeze(0).repeat(a.shape[0], 1) # (N,3)
        v = torch.cross(a, b, dim=1) # (N,3)
        s = torch.norm(v, dim=1) # (N,)
        c = torch.sum(a * b, dim=1) # (N,)
        mask = s < 1e-8 # (N,)
        R = torch.eye(3, device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1)
        vx = torch.tensor([[    0, -v[:,2],  v[:,1]],
                            [ v[:,2],     0, -v[:,0]],
                            [-v[:,1],  v[:,0],     0]], device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1)
        R[~mask] = torch.eye(3, device=a.device).unsqueeze(0).repeat(a.shape[0], 1, 1) + vx[~mask] + vx[~mask] @ vx[~mask] * ((1 - c[~mask]) / (s[~mask]**2)).unsqueeze(-1).unsqueeze(-1)
        return R

    def compute_angular_integration(self, hand_gaussians):
        """
        Compute the integrated black-box function over the angular extent of all Gaussians.
        
        Args:
            n: A 3D normal vector (torch tensor of shape (3,)) at point x.
            f: A black-box function that takes (theta, phi) tensors (each of shape (N,))
               and returns a tensor of the same shape.
        
        Returns:
            A scalar (torch tensor) equal to the sum over Gaussians of:
                f(theta_max, phi_max) - f(theta_min, phi_max)
              - f(theta_max, phi_min) + f(theta_min, phi_min)
            where for each Gaussian the angular bounds are computed as:
              - mean_azimuth, mean_elevation (from the rotated center)
              - half_azimuth = arctan(a_xy / d)
              - half_elevation = arctan(a_z / d)
              and then:
                 theta_min = mean_azimuth - half_azimuth,
                 theta_max = mean_azimuth + half_azimuth,
                 phi_min   = mean_elevation - half_elevation,
                 phi_max   = mean_elevation + half_elevation.
        """
        device = self._rotation.device

        # Step 1. Compute R_align: rotation that sends n to z-axis.
        normals = self.get_normal()  # (N,3)
        z = torch.tensor([0.0, 0.0, 1.0], device=device)

        totals = []
        lamda = np.zeros(normals.shape[0])
        C0 = 0.28209479177387814

        pbar = tqdm(range(normals.shape[0]))
        for i in pbar:
            pt = self._xyz[i] # (3,)
            n = normals[i] # (3,)
            R_align = self._rotation_matrix_from_vectors(n, z)  # (3,3)
            print (R_align.shape)
            exit()

            # Step 2. Rotate the centers (_xyz) to get their directions in the aligned frame.
            # self._xyz is assumed to have shape (M, 3)
            centers = hand_gaussians._xyz  # (M,3)
            # Rotate each center: x_rot = R_align @ center, done for all Gaussians.
            # Use torch.matmul: (3,3) @ (3,M) then transpose.
            centers_rot = torch.matmul(R_align, (centers).T).T  # (M,3)
            pt_rot = torch.matmul(R_align, pt)  # (3,)
            centers_rot = centers_rot - pt_rot

            # Compute distance d and then spherical coordinates:
            d = torch.norm(centers_rot, dim=1)  # (M,)
            # Mean azimuth: arctan2(z, y)
            mean_azimuth = torch.atan2(centers_rot[:,2], centers_rot[:,1])  # (M,)
            # Mean elevation: arctan2(z, x)
            mean_elevation = torch.atan2(centers_rot[:,2], centers_rot[:,0])  # (M,)

            # Step 3. Compute effective extents from rotation and scaling.
            # Get per-Gaussian rotation matrices: R_i from _rotation.
            R_i = build_rotation(hand_gaussians._rotation)  # shape (M, 3, 3)
            # Get scaling from self.get_scaling: shape (M, 3)
            s = hand_gaussians.get_scaling  # (M,3)
            # Form diagonal matrices: S with shape (M,3,3)
            S = torch.diag_embed(s)
            # Effective transformation: M_eff = R_align @ (R_i @ S).
            # R_align has shape (3,3); we expand it for all Gaussians.
            M_eff = torch.matmul(R_align.unsqueeze(0), torch.bmm(R_i, S))  # (M,3,3)

            # For the effective extents, we extract:
            # a_xz: maximum projection of each column of M_eff onto the x-z plane.
            xz_components = M_eff[:, [0, 2], :]   # shape (M,2,3)
            xz_norms = torch.norm(xz_components, dim=1)  # shape (M,3) - norm of each column.
            a_xz, _ = torch.max(xz_norms, dim=1)  # (M,)

            # a_yz: maximum projection of each column of M_eff onto the x-z plane.
            yz_components = M_eff[:, [1, 2], :]   # shape (M,2,3)
            yz_norms = torch.norm(yz_components, dim=1)  # shape (M,3) - norm of each column.
            a_yz, _ = torch.max(yz_norms, dim=1)  # (M,)

            lamda = torch.norm(s, dim=1) # (M,)

            # Step 4. Compute half-angular extents.
            # Since the projection of a Gaussian center at distance d,
            # half angular extent = arctan(effective_extent / d)
            half_azimuth = torch.atan(a_yz / d)    # (M,)
            half_elevation = torch.atan(a_xz / d)     # (M,)

            # Step 5. Compute the angular bounds for each Gaussian.
            theta_min = mean_azimuth - half_azimuth  # (M,)
            theta_max = mean_azimuth + half_azimuth  # (M,)
            phi_min   = mean_elevation - half_elevation  # (M,)
            phi_max   = mean_elevation + half_elevation  # (M,)

            theta_max = torch.clamp(theta_max, 0, np.pi)
            theta_min = torch.clamp(theta_min, 0, np.pi)
            phi_max = torch.clamp(phi_max, 0, np.pi)
            phi_min = torch.clamp(phi_min, 0, np.pi)

            # Step 6. Evaluate the black-box function f at the four corners.
            val = get_frac(theta_max, phi_max, lamda) - get_frac(theta_min, phi_max, lamda) - get_frac(theta_max, phi_min, lamda) + get_frac(theta_min, phi_min, lamda)
            # Sum the contributions from all Gaussians.
            total = torch.sum(val)
            # return total
            total = torch.clamp(total, 0.0, 1.0)
            totals.append(total.item())
            self._features_dc[i] = ((self._features_dc[i] * C0 + 0.5) * (1-total) - 0.5) / 0.28209479177387814
            # self._features_rest[i] = self._features_rest[i] * (1-total)

            # self._features_dc[i] = ((self._features_dc[i] * C0 + 0.5) * 0.0 - 0.5) / 0.28209479177387814
            # self._features_rest[i] = 

        # totals = np.array(totals)
        # print ("Max: ", np.max(totals), "Min: ", np.min(totals), "Mean: ", np.mean(totals), "Std: ", np.std(totals))
        # print (lamda.min(), lamda.max())
