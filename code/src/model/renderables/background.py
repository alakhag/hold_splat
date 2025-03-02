import torch
import torch.nn as nn
from src.model.renderables.node import Splats

class BackgroundSplats(Splats):
    def __init__(self, seq_name, node_id, num_frames):
        class_id = 0
        super(ObjectSplats, self).__init__(node_id=node_id, class_id=class_id)
        self.load_pcd()
        self.gen_from_pcd(num_frames)

    def deform(self, input):
        x_c = self._xyz.unsqueeze(0)
        rots = self._rotation.unsqueeze(0)

        return x_c, rots
