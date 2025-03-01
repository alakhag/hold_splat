import numpy as np
import torch
from kaolin.ops.mesh import index_vertices_by_faces

import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node, Splats
from src.datasets.utils import get_camera_params
from common.body_models import seal_mano_v
from common.body_models import seal_mano_f
from src.utils.meshing import generate_mesh
from src.model.mano.deformer import MANODeformer
from src.model.mano.server import MANOServer
import src.hold.hold_utils as hold_utils



def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to rotation matrices.
    
    Args:
        quat (torch.Tensor): shape (N, 4) where each quaternion is (w, x, y, z).
        
    Returns:
        torch.Tensor: Rotation matrices of shape (N, 3, 3).
    """
    # Normalize the quaternions to avoid scaling issues.
    quat = quat / quat.norm(dim=1, keepdim=True)
    w, x, y, z = quat.unbind(dim=1)  # each has shape (N,)
    
    # Compute the rotation matrix elements.
    N = quat.shape[0]
    rotmat = torch.stack([
        1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)
    ], dim=1).reshape(N, 3, 3)
    return rotmat

def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to quaternions.
    
    Args:
        rotmat (torch.Tensor): shape (B, 3, 3)
        
    Returns:
        torch.Tensor: Quaternions of shape (B, 4) in (w, x, y, z) order.
    """
    B = rotmat.shape[0]
    quat = torch.empty(B, 4, device=rotmat.device, dtype=rotmat.dtype)
    
    # Calculate trace of each matrix
    m00 = rotmat[:, 0, 0]
    m11 = rotmat[:, 1, 1]
    m22 = rotmat[:, 2, 2]
    trace = m00 + m11 + m22

    # Case 1: trace > 0
    pos_mask = trace > 0
    if pos_mask.any():
        s = torch.sqrt(trace[pos_mask] + 1.0) * 2  # s = 4 * w
        quat[pos_mask, 0] = 0.25 * s
        quat[pos_mask, 1] = (rotmat[pos_mask, 2, 1] - rotmat[pos_mask, 1, 2]) / s
        quat[pos_mask, 2] = (rotmat[pos_mask, 0, 2] - rotmat[pos_mask, 2, 0]) / s
        quat[pos_mask, 3] = (rotmat[pos_mask, 1, 0] - rotmat[pos_mask, 0, 1]) / s

    # Case 2: trace <= 0
    neg_mask = ~pos_mask
    if neg_mask.any():
        # For these samples, we choose the largest diagonal element.
        m0 = rotmat[neg_mask, 0, 0]
        m1 = rotmat[neg_mask, 1, 1]
        m2 = rotmat[neg_mask, 2, 2]
        # Get indices in the batch that satisfy the negative mask
        idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        
        for i in idx:
            r = rotmat[i]
            if r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
                s = torch.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
                quat[i, 0] = (r[2, 1] - r[1, 2]) / s
                quat[i, 1] = 0.25 * s
                quat[i, 2] = (r[0, 1] + r[1, 0]) / s
                quat[i, 3] = (r[0, 2] + r[2, 0]) / s
            elif r[1, 1] >= r[0, 0] and r[1, 1] >= r[2, 2]:
                s = torch.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
                quat[i, 0] = (r[0, 2] - r[2, 0]) / s
                quat[i, 1] = (r[0, 1] + r[1, 0]) / s
                quat[i, 2] = 0.25 * s
                quat[i, 3] = (r[1, 2] + r[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
                quat[i, 0] = (r[1, 0] - r[0, 1]) / s
                quat[i, 1] = (r[0, 2] + r[2, 0]) / s
                quat[i, 2] = (r[1, 2] + r[2, 1]) / s
                quat[i, 3] = 0.25 * s

    # Normalize the resulting quaternion to ensure unit norm.
    quat = quat / quat.norm(dim=1, keepdim=True)
    return quat

class MANONode(Node):
    def __init__(self, args, opt, betas, sdf_bounding_sphere, node_id):
        if node_id == "right":
            class_id = 2
            self.is_rhand = True
        elif node_id == "left":
            class_id = 3
            self.is_rhand = False
        else:
            assert False

        deformer = MANODeformer(max_dist=0.1, K=15, betas=betas, is_rhand=self.is_rhand)
        server = MANOServer(betas=betas, is_rhand=self.is_rhand)

        from src.model.mano.params import MANOParams
        from src.model.mano.specs import mano_specs

        params = MANOParams(
            args.n_images,
            {
                "betas": 10,
                "global_orient": 3,
                "transl": 3,
                "pose": 45,
            },
            node_id,
        )
        params.load_params(args.case)
        super(MANONode, self).__init__(
            args,
            opt,
            mano_specs,
            sdf_bounding_sphere,
            opt.implicit_network,
            opt.rendering_network,
            deformer,
            server,
            class_id,
            node_id,
            params,
        )

        self.mesh_v_cano = self.server.verts_c
        self.mesh_f_cano = torch.tensor(
            self.server.human_layer.faces.astype(np.int64)
        ).cuda()
        self.mesh_face_vertices = index_vertices_by_faces(
            self.mesh_v_cano, self.mesh_f_cano
        )

        self.mesh_v_cano_div = None
        self.mesh_f_cano_div = None
        self.canonical_mesh = None

    def sample_points(self, input):
        node_id = self.node_id
        full_pose = input[f"{node_id}.full_pose"]
        output = self.server(
            input[f"{node_id}.params"][:, 0],
            input[f"{node_id}.transl"],
            full_pose,
            input[f"{node_id}.betas"],
        )

        debug.debug_world2pix(self.args, output, input, self.node_id)
        cond = {"pose": full_pose[:, 3:] / np.pi}  # pose-dependent shape
        if self.training:
            if input["current_epoch"] < 20:
                cond = {"pose": full_pose[:, 3:] * 0.0}  # no pose for shape

        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        deform_info = {
            "cond": cond,
            "tfs": output["tfs"],
            "verts": output["verts"],
        }
        z_vals = self.ray_sampler.get_z_vals(
            volsdf_utils.sdf_func_with_deformer,
            self.deformer,
            self.implicit_network,
            ray_dirs,
            cam_loc,
            self.density,
            self.training,
            deform_info,
        )

        # fg samples to points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        out = {}
        out["idx"] = input["idx"]
        out["output"] = output
        out["cond"] = cond
        out["ray_dirs"] = ray_dirs
        out["cam_loc"] = cam_loc
        out["deform_info"] = deform_info
        out["z_vals"] = z_vals
        out["points"] = points
        out["tfs"] = output["tfs"]
        out["batch_size"] = batch_size
        out["num_pixels"] = num_pixels
        return out

    def spawn_cano_mano(self, sample_dict_h):
        mesh_v_cano = sample_dict_h["output"]["v_posed"]
        mesh_vh_cano = seal_mano_v(mesh_v_cano)
        mesh_fh_cano = seal_mano_f(self.mesh_f_cano, self.is_rhand)

        mesh_vh_cano, mesh_fh_cano = hold_utils.subdivide_cano(
            mesh_vh_cano, mesh_fh_cano
        )
        self.mesh_v_cano_div = mesh_vh_cano
        self.mesh_f_cano_div = mesh_fh_cano

    def meshing_cano(self, pose=None):
        if pose is None:
            cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}
        else:
            cond = {"pose": pose / np.pi}
        assert cond["pose"].shape[0] == 1, "only support batch size 1"
        v_min_max = np.array([[-0.0814, -0.0280, -0.0742], [0.1171, 0.0349, 0.0971]])
        mesh_canonical = generate_mesh(
            lambda x: hold_utils.query_oc(self.implicit_network, x, cond),
            v_min_max,
            point_batch=10000,
            res_up=1,
            res_init=64,
        )
        return mesh_canonical

class MANOSplats(Splats):
    def __init__(self, betas, node_id, num_frames):
        if node_id == "right":
            class_id = 2
            self.is_rhand = True
        elif node_id == "left":
            class_id = 3
            self.is_rhand = False
        else:
            assert False

        deformer = MANODeformer(max_dist=0.1, K=15, betas=betas, is_rhand=self.is_rhand)
        server = MANOServer(betas=betas, is_rhand=self.is_rhand)
        super(MANOSplats, self).__init__(deformer=deformer, server=server, node_id=node_id, class_id=class_id)
        self.load_pcd()
        self.gen_from_pcd(num_frames)

    def deform(self, input):
        full_pose = input[f"{node_id}.full_pose"]
        output = self.server(
            1.0,
            input[f"{node_id}.transl"],
            full_pose,
            input[f"{node_id}.betas"],
        )

        cond = {"pose": full_pose[:, 3:] / np.pi}  # pose-dependent shape
        if self.training:
            if input["current_epoch"] < 20:
                cond = {"pose": full_pose[:, 3:] * 0.0}  # no pose for shape

        tfs = output["tfs"]
        x_c = self._xyz.unsqueeze(0).expand(tfs.shape[0], -1, -1)
        x, _, T = self.deformer(x_c, tfs, return_weights=False, inverse=False, verts=None, return_tfs=True)
        rots = self._rotation
        base_rots_mat = quat_to_rotmat(rots)  # (N, 3, 3)
        base_rots_mat = base_rots_mat.unsqueeze(0).expand(B, -1, -1, -1)
        T_rot = T[:, :, :3, :3]
        new_rots_mat = torch.matmul(T_rot, base_rots_mat)  # (B, N, 3, 3)
        new_rots = rotmat_to_quat(new_rots_mat)  # (B, N, 4)

        return x, new_rots

