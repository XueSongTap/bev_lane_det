import numpy as np
import copy
import time
import sys
sys.path.append('/workspace/bev_lane_det')
import os
gpu_id = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_id])
import json
from torch.utils.data import Dataset,DataLoader
import copy
import time
from tqdm import tqdm
from utils.config_util import load_config_module
from models.util.load_model import load_model
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from utils.util_val.val_offical import LaneEval
from models.model.single_camera_bev import *

model_path = '/dataset/model/20220814/ep008.pth' #model path of verification

''' parameter from config '''
config_file = './openlane_config.py'
configs = load_config_module(config_file)
gt_paths = configs.val_gt_paths
image_paths = configs.val_image_paths
x_range = configs.x_range
y_range = configs.y_range
meter_per_pixel = configs.meter_per_pixel


'''Post-processing parameters '''
post_conf = -0.7 # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 15 # The minimum number of points in a cluster

tmp_save_path = '/dataset/tmp/tmp_openlane' #tmp path for save intermediate result


class PostProcessDataset(Dataset):
    def __init__(self, model_res_save_path, postprocess_save_path, gt_paths):
        self.valid_data = os.listdir(model_res_save_path)
        self.postprocess_save_path = postprocess_save_path
        self.model_res_save_path = model_res_save_path
        self.gt_paths = gt_paths
        self.x_range = x_range
        self.meter_per_pixel = meter_per_pixel

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, item):
        loaded = np.load(os.path.join(self.model_res_save_path, self.valid_data[item]))
        prediction = (loaded[:, 0:1, :, :], loaded[:, 1:3, :, :])
        offset_y = loaded[:, 3:4, :, :][0][0]
        z_pred = loaded[:, 4:5, :, :][0][0]
        files = self.valid_data[item].split('.')[0].split('__')
        gt_path = os.path.join(self.gt_paths, files[0], files[1] + '.json')
        gt, matrix_ours2persformer = self.get_ego_by_cnt_persformer(gt_path)
        canvas, ids = embedding_post(prediction, conf=post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
        lines = bev_instance2points_with_offset_z(canvas, max_x=self.x_range[1],
                                                  meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
                                                  offset_y=offset_y, Z=z_pred)
        frame_lanes_pred = []
        for lane in lines:
            pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            frame_lanes_pred.append(pred_in_persformer.T.tolist())
        frame_lanes_gt = []
        for lane in gt:
            frame_lanes_gt.append(lane[:3].T.tolist())
        with open(os.path.join(self.postprocess_save_path, files[1] + '.json'), 'w') as f1:
            json.dump([frame_lanes_pred, frame_lanes_gt], f1)
        return torch.zeros((3, 3)) # a random number


    def get_ego_by_cnt_persformer(self,gt_path):
        cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]], dtype=float))

        R_vg = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=float)
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        cam_w_extrinsics = np.array(gt['extrinsic'])
        cam_extrinsics_persformer = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_persformer[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_persformer[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_persformer[0:2, 3] = 0.0
        matrix_lane2persformer = cam_extrinsics_persformer @ cam_representation
        lanes = gt['lane_lines']
        frame_lanes = []
        for idx in range(len(lanes)):
            lane1 = lanes[idx]
            lane_camera_w = np.array(lane1['xyz']).T[np.array(lane1['visibility']) == 1.0].T
            lane_camera_w = np.vstack((lane_camera_w, np.ones((1, lane_camera_w.shape[1]))))
            lane_ego_persformer = matrix_lane2persformer @ lane_camera_w  #
            distance = np.sqrt((lane_ego_persformer[1][0] - lane_ego_persformer[1][-1]) ** 2 + (
                    lane_ego_persformer[0][0] - lane_ego_persformer[0][-1]) ** 2)
            if distance < 3:
                continue
            frame_lanes.append(lane_ego_persformer)
        matrix_ours2persformer = matrix_lane2persformer @ np.linalg.inv(cam_w_extrinsics)
        return frame_lanes, matrix_ours2persformer


def val():
    model = configs.model()
    model = load_model(model,
                       model_path)
    print(model_path)
    model.cuda()
    model.eval()
    val_dataset = configs.val_dataset()
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=16,
                              num_workers=8,
                              shuffle=False)
    ''' Make temporary storage files according to time '''
    time1 = int(time.time()) # 
    np_save_path = os.path.join(tmp_save_path, str(time1) + '_np')
    os.makedirs(np_save_path, exist_ok=True)
    res_save_path = os.path.join(tmp_save_path, str(time1) + '_result')
    os.makedirs(res_save_path, exist_ok=True)
    ''' get model result and save'''
    for item in tqdm(val_loader):
        image,bn_name = item
        image = image.cuda()
        with torch.no_grad():
            pred_ = model(image)[0]
            seg = pred_[0].detach().cpu()
            embedding = pred_[1].detach().cpu()
            offset_y = torch.sigmoid(pred_[2]).detach().cpu()
            z_pred = pred_[3].detach().cpu()
            for idx in range(seg.shape[0]):
                ms, me, moffset, z = seg[idx].unsqueeze(0).numpy(), embedding[idx].unsqueeze(0).numpy(), offset_y[
                    idx].unsqueeze(0).numpy(), z_pred[idx].unsqueeze(0).numpy()
                tmp_res_for_save = np.concatenate((ms, me, moffset, z), axis=1)
                save_path = os.path.join(np_save_path,
                                         bn_name[0][idx] + '__' + bn_name[1][idx].replace('json', 'np'))
                np.save(save_path, tmp_res_for_save)
    ''' get postprocess result and save '''
    postprocess = PostProcessDataset(np_save_path, res_save_path, gt_paths)
    postprocess_loader = DataLoader(dataset=postprocess,
                                    batch_size=32,
                                    num_workers=16,
                                    shuffle=False)
    for item in tqdm(postprocess_loader):
        continue
    ''' verification by official tools '''
    lane_eval = LaneEval()
    res_list = os.listdir(res_save_path)
    for item in tqdm(res_list):
        with open(os.path.join(res_save_path, item), 'r') as f:
            res = json.load(f)
        lane_eval.bench_all(res[0], res[1])
    lane_eval.show()

if __name__ == '__main__':
    val()
