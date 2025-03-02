import sys

import torch
import torch.nn as nn
from loguru import logger


from src.model.renderables.background import BackgroundSplats
from src.model.renderables.object_node import ObjectSplats
from src.model.renderables.mano_node import MANOSplats
from src.utils.eval_sh import RGB2SH, SH2RGB

sys.path = [".."] + sys.path
from common.xdict import xdict

from gaussian_renderer import render
from copy import deepcopy

class SplatNet(nn.Module):
    def __init__(
        self,
        betas_r,
        betas_l,
        num_frames,
        args,
    ):
        super(SplatNet, self).__init__()
        self.args = args
        self.threshold = 0.05
        node_dict = {}
        if betas_r is not None:
            right_node = MANOSplats(betas_r, "right", args.n_images, args.case)
            node_dict["right"] = right_node

        if betas_l is not None:
            left_node = MANOSplats(betas_l, "left", args.n_images, args.case)
            node_dict["left"] = left_node

        object_node = ObjectSplats(args.case, "object", args.n_images)
        node_dict["object"] = object_node
        bg_node = BackgroundSplats(args.case, "bg", args.n_images)
        node_dict["bg"] = bg_node
        self.nodes = nn.ModuleDict(node_dict)

    def forward(self, input):
        input = xdict(input)
        out_dict = xdict()
        if self.training:
            out_dict["epoch"] = input["current_epoch"]
            out_dict["step"] = input["global_step"]

        torch.set_grad_enabled(True)
        sample_dict = None
        factors_dicts = {}
        
        for node in self.nodes.values():
            factors = self.nodes[node.node_id](input)
            factors_dicts[node.node_id] = factors
        
        factors_dicts = self.merge(factors_dicts)

        rgb = self.render(factors_dicts["rgb"], override=False)
        sem = self.render(factors_dicts["mask"], override=True)

    def get_mask_gs(self, _dict, node_id):
        features_dc = _dict["features"][:,:,0:1,:]
        mask_dc = torch.zeros_like(features_dc)
        if node_id=="right":
            mask_dc[...,2] = 1.0
        elif node_id=="object":
            mask_dc[...,1] = 1.0
        out["override"] = mask_dc

    def process(self, factors_dict):
        return {
            "xyz": factors_dict[0],
            "rots": factors_dict[1],
            "opacity": factors_dict[2],
            "scale": factors_dict[3],
            "features": factors_dict[4]
        }

    def merge(self, factors_dicts):
        right_dict = self.process(factors_dicts["right"])
        object_dict = self.process(factors_dicts["object"])
        bg_dict = self.process(factors_dicts["bg"])

        right_dict = self.get_mask_gs(right_dict, "right")
        object_dict = self.get_mask_gs(object_dict, "object")
        bg_dict = self.get_mask_gs(bg_dict, "bg")

        fg_dict = self.composite(right_dict, object_dict)
        full_dict = self.composite(fg_dict, bg_dict)

        return {
            "right": right_dict,
            "object": object_dict,
            "bg": bg_dict,
            "fg": fg_dict,
            "rgb": full_dict,
            "mask": mask_dict
        }

    def composite(self, dict1, dict2):
        out = {}
        keys = dict1.keys()
        for k in keys:
            item1 = dict1[k]
            item2 = dict2[k]
            batch_size = item1.shape[0]
            if item2.shape[0]==1:
                l = len(item2.shape)
                expand_shape = [-1]*l
                expand_shape[0] = batch_size
                item2 = item2.expand(expand_shape)
            item = torch.concatenate([item1, item2], dim=1)
            out[k] = item
        return out

    def render(self, _dict, override=False):
        xyz = _dict["xyz"]
        xyz = xyz.reshape(-1, *xyz.shape[2:])
        opacity = _dict["opacity"]
        opacity = opacity.reshape(-1, *xyz.shape[2:])
        scale = _dict["scale"]
        scale = scale.reshape(-1, *xyz.shape[2:])
        rots = _dict["rots"]
        rots = rots.reshape(-1, *xyz.shape[2:])
        if override:
            precomp = _dict["override"]
            precomp = precomp.reshape(-1, *xyz.shape[2:])
            return render("viewpoint_camera: FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform, camera_center", 
                {xyz, opacity, scale, rots, None}, 
                torch.tensor([0,0,0], dtype=torch.float32, device="cuda"), 
                override_color=precomp, 
                use_trained_exp=False)
        else:
            features = _dict["features"]
            features = features.reshape(-1, *features.shape[2:])
            return render("viewpoint_camera", {xyz, opacity, scale, rots, features}, torch.tensor([0,0,0], dtype=torch.float32, device="cuda"), override_color=None, use_trained_exp=False)

