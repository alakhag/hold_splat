import numpy as np
import torch

from src.model.renderables.node import Splats
from src.model.mano.deformer import MANODeformer
from src.model.mano.server import MANOServer



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

class MANOSplats(Splats):
    def __init__(self, betas, node_id, num_frames, seq_name):
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

        params = MANOParams(
            num_frames,
            {
                "betas": 10,
                "global_orient": 3,
                "transl": 3,
                "pose": 45,
            },
            node_id,
        )
        params.load_params(seq_name)
        super(MANOSplats, self).__init__(deformer=deformer, server=server, node_id=node_id, class_id=class_id, params=params)
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

