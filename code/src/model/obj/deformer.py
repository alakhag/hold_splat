import torch
from sklearn.neighbors import NearestNeighbors

def knn_points_sklearn(pts, verts, K):
    B, N, _ = pts.shape
    distances_list, indices_list, neighbor_points_list = [], [], []

    for b in range(B):
        knn = NearestNeighbors(n_neighbors=K, algorithm='auto')
        knn.fit(verts[b].detach().cpu().numpy())  # Fit the reference points
        distances, indices = knn.kneighbors(pts[b].detach().cpu().numpy())  # Query neighbors

        distances_list.append(torch.tensor(distances, dtype=torch.float32))
        indices_list.append(torch.tensor(indices, dtype=torch.long))
        neighbor_points_list.append(torch.tensor(verts[b].detach().cpu().numpy()[indices], dtype=torch.float32))

    distance_batch = torch.stack(distances_list).to(pts.device)
    index_batch = torch.stack(indices_list).to(pts.device)
    neighbor_points = torch.stack(neighbor_points_list).to(pts.device)

    return distance_batch, index_batch, neighbor_points


class ObjectDeformer:
    def __init__(self):
        super().__init__()
        self.max_dist = 0.1

    def forward(self, x, tfs, return_weights=None, inverse=False, verts=None):
        """
        tfs: (batch, 4, 4)
        x: (batch, N, 3)
        """
        assert len(x.shape) == 3
        assert x.shape[2] == 3
        tfs = tfs.view(-1, 4, 4)

        # inverse: deform -> cano
        # not inverse: cano -> deform
        if inverse:
            obj_tfs = torch.inverse(tfs)
        else:
            obj_tfs = tfs

        # apply transformation
        x_pad = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1).permute(
            0, 2, 1
        )  # (N, 4)
        obj_x_tf = torch.bmm(obj_tfs, x_pad).permute(0, 2, 1)
        x_tf = obj_x_tf[:, :, :3]
        outlier_mask = None
        if verts is not None and inverse:  # points in deform space
            distance_batch, index_batch, neighbor_points = knn_points_sklearn(
                x, verts, K=1
            )
            distance_batch = torch.clamp(distance_batch, max=4)
            distance_batch = torch.sqrt(distance_batch)
            distance_batch = distance_batch.min(dim=2).values
            outlier_mask = distance_batch > self.max_dist
        return x_tf, outlier_mask

    def forward_skinning(self, xc, cond, tfs):
        # cano -> deform
        x_transformed = self.forward(xc, tfs, inverse=False)[0]
        return x_transformed
