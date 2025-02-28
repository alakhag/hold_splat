# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.data.datasets.pascal_voc import register_pascal_voc

_datasets_root = "data"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')
import os.path as op
from glob import glob

def main(seq_name, box_scale, min_size, max_size, vis_raw):
    # load cfg and model
    cfg = get_cfg()
    cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model

    # data path
    seq_folder = f'../data/{seq_name}'
    fnames = sorted(glob(op.join(seq_folder, 'images', '*')))


    # predict
    predictor = DefaultPredictor(cfg)
    boxes_all = []
    # output
    from tqdm import tqdm
    prev_outputs = None
    for fname in tqdm(fnames):
        out_p = fname.replace('/images/', '/processed/crop_image/')
        os.makedirs(op.dirname(out_p), exist_ok=True)
        im = cv2.imread(fname)
        outputs = predictor(im)
        
        if vis_raw:
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(out_p, v.get_image()[:, :, ::-1])

        boxes = outputs["instances"].pred_boxes        
        import copy
        if len(boxes) == 0:
            outputs = copy.deepcopy(prev_outputs)
            boxes = outputs["instances"].pred_boxes    
            box_scale = 2.5
        else:
            prev_outputs = copy.deepcopy(outputs)
            
        from PIL import Image
        im = Image.open(fname)
        boxes_list = [] 
        for box in boxes:
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            boxes_list.append(area)

        idx = np.argmax(boxes_list)
        box = boxes.tensor.tolist()[idx]
        score = float(outputs["instances"].scores[idx])
        cls = int(outputs["instances"].pred_classes[idx])

        # crop
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        size = min(max(max(w, h)*box_scale, min_size), max_size)
        
        box_new = [cx-size/2, cy-size/2, cx+size/2, cy+size/2]
        if not vis_raw:
            im = im.crop(box_new)
            im.save(out_p)
        # box_new = box_new + [score, cls]
        boxes_all.append(box_new)
    boxes_all = np.array(boxes_all)
    out_p = op.join(seq_folder, 'processed', "boxes.npy")
    np.save(out_p, boxes_all)
    print(f"Saved to {out_p}")
        
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, default='')
    parser.add_argument('--scale', type=float)
    parser.add_argument('--min_size', type=float, default=256)
    parser.add_argument('--max_size', type=float, default=float('inf'))
    parser.add_argument('--vis_raw', action='store_true')
    args = parser.parse_args()
    
    main(seq_name=args.seq_name, box_scale=args.scale, min_size=args.min_size, max_size=args.max_size, vis_raw=args.vis_raw)

