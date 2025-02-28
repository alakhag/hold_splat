from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel
from tqdm import tqdm

def visualize_2d(results_2d):
    from PIL import Image
    import matplotlib.pyplot as plt
    
    j2d_r = results_2d['j2d.right']
    j2d_l = results_2d['j2d.left']

    v2d_r = results_2d['v2d.right']
    v2d_l = results_2d['v2d.left']

    im_paths = results_2d['im_paths']

    print("Visualizing 2D keypoints")
    for idx in tqdm(range(len(im_paths))):

        im_p = im_paths[idx]
        out_p = im_p.replace("/images/", '/processed/2d_keypoints/')

        im = Image.open(im_p)

        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.scatter(j2d_r[idx, :, 0], j2d_r[idx, :, 1], s=10)
        plt.scatter(j2d_l[idx, :, 0], j2d_l[idx, :, 1], s=10)
        plt.legend(['jts_r', 'jts_l'])
        plt.savefig(out_p)
        plt.close()

    print("Visualizing 2D vertices")
    for idx in tqdm(range(len(im_paths))):

        im_p = im_paths[idx]
        out_p = im_p.replace("/images/", '/processed/hpe_vis/')

        im = Image.open(im_p)

        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.scatter(v2d_r[idx, :, 0], v2d_r[idx, :, 1], s=1)
        plt.scatter(v2d_l[idx, :, 0], v2d_l[idx, :, 1], s=1)
        plt.legend(['mano_r', 'mano_l'])
        plt.savefig(out_p)
        plt.close()


def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    zz = x_homo[:, :, 2:3]
    
    
    x = x_homo[:, :, :2] / zz
    return x


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d


def reform_pred_list(pred_list):
    im_paths = sorted(list(set([pred_dict['img_path'] for pred_dict in pred_list])))

    verts_r = np.zeros((len(im_paths), 778, 3))*np.nan
    verts_l = np.copy(verts_r)
    
    joints_r = np.zeros((len(im_paths), 21, 3))*np.nan
    joints_l = np.copy(joints_r)


    for pred_dict in pred_list:
        is_right = bool(pred_dict['is_right'])

        v3d_cam = pred_dict['verts']  + pred_dict['cam_t.full'][None, :]
        j3d_cam = pred_dict['jts']  + pred_dict['cam_t.full'][None, :]

        idx = im_paths.index(pred_dict['img_path'])

        if is_right:
            verts_r[idx] = v3d_cam
            joints_r[idx] = j3d_cam
        else:
            verts_l[idx] = v3d_cam
            joints_l[idx] = j3d_cam

    verts_r = verts_r.astype(np.float32)
    verts_l = verts_l.astype(np.float32)
    joints_r = joints_r.astype(np.float32)
    joints_l = joints_l.astype(np.float32)
    
    
    K = torch.FloatTensor(pred_list[0]['K'])
    joints_r = torch.FloatTensor(joints_r)
    joints_l = torch.FloatTensor(joints_l)
    verts_r = torch.FloatTensor(verts_r)
    verts_l = torch.FloatTensor(verts_l)
    
    v2d_r = project2d_batch(K[None, :, :].repeat(verts_r.shape[0], 1, 1), verts_r).numpy()
    v2d_l = project2d_batch(K[None, :, :].repeat(verts_l.shape[0], 1, 1), verts_l).numpy()
    j2d_r = project2d_batch(K[None, :, :].repeat(joints_r.shape[0], 1, 1), joints_r).numpy()
    j2d_l = project2d_batch(K[None, :, :].repeat(joints_l.shape[0], 1, 1), joints_l).numpy()    

    results_3d = {}
    results_3d['v3d.right'] = verts_r
    results_3d['v3d.left'] = verts_l
    results_3d['j3d.right'] = joints_r
    results_3d['j3d.left'] = joints_l
    results_3d['im_paths'] = im_paths
    results_3d['K'] = pred_list[0]['K']
    
    results_2d = {}
    results_2d['v2d.right'] = v2d_r
    results_2d['v2d.left'] = v2d_l
    results_2d['j2d.right'] = j2d_r
    results_2d['j2d.left'] = j2d_l
    results_2d['im_paths'] = im_paths
    
    return results_3d, results_2d



import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--seq_name', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
    
    args.img_folder = f'../data/{args.seq_name}/images'

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    assert len(img_paths) > 0, f"No images found in {args.img_folder}"

    # Iterate over all images in folder
    print('Running inference on images')
    pred_list = []
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                jts = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                jts[:,0] = (2*is_right-1)*jts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                pred_dict = {}
                pred_dict['cam_t.full'] = cam_t
                pred_dict['verts'] = verts
                pred_dict['jts'] = jts
                pred_dict['is_right'] = is_right
                pred_dict['img_path'] = str(img_path)

                fx = fy = float(scaled_focal_length.cpu().numpy())
                cx, cy = img_size[n].cpu().detach().numpy() / 2
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                pred_dict['K'] = K
                pred_list.append(pred_dict)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    import os.path as op
    out_3d_p = op.join(args.img_folder, '../processed/v3d.npy')
    out_2d_p = op.join(args.img_folder, '../processed/j2d.full.npy')
    # normalize paths
    out_3d_p = op.normpath(out_3d_p)
    out_2d_p = op.normpath(out_2d_p)
    os.makedirs(op.dirname(out_3d_p), exist_ok=True)
    results_3d, results_2d = reform_pred_list(pred_list)
    visualize_2d(results_2d)
    np.save(out_3d_p, results_3d)
    np.save(out_2d_p, results_2d)
    print(f"Saved 3D results to {out_3d_p}")
    print(f"Saved 2D results to {out_2d_p}")

if __name__ == '__main__':
    main()
