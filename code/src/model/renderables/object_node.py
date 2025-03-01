import torch
import numpy as np
import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node, Splats
from src.datasets.utils import get_camera_params
from src.utils.meshing import generate_mesh
import torch.nn as nn
from src.model.obj.deformer import ObjectDeformer
from src.model.obj.server import ObjectServer


class ObjectSplats(Splats):
    def __init__(self, seq_name, node_id, num_frames):
        deformer = ObjectDeformer()
        server = ObjectServer(seq_name, None)
        class_id = 1
        super(ObjectSplats, self).__init__(deformer=deformer, server=server, node_id=node_id, class_id=class_id)
        self.load_pcd()
        self.gen_from_pcd(num_frames)

    def deform(self, input):
        scene_scale = 1.0
        obj_pose = input[f"{node_id}.global_orient"]
        obj_trans = input[f"{node_id}.transl"]
        obj_output = self.server(scene_scale, obj_trans, obj_pose)

        tfs = obj_output["obj_tfs"][:, 0]
        x_c = self._xyz.unsqueeze(0).expand(tfs.shape[0], -1, -1)
        x, _ = self.deformer(x_c, tfs, return_weights=False, inverse=False, verts=None)
        rots = self._rotation
        base_rots_mat = quat_to_rotmat(rots)  # (N, 3, 3)
        base_rots_mat = base_rots_mat.unsqueeze(0).expand(B, -1, -1, -1)
        tfs_rot = tfs[:, :3, :3]
        new_rots_mat = torch.matmul(tfs_rot.unsqueeze(1), base_rots_mat)  # (B, N, 3, 3)
        new_rots = rotmat_to_quat(new_rots_mat)  # (B, N, 4)

        return x, new_rots
    
