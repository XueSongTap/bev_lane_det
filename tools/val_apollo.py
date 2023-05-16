import os
gpu_id = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_id])
import sys
sys.path.append('/workspace/bev_lane_det')
import shutil
import numpy as np
import json
import time
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.config_util import load_config_module
from models.util.load_model import load_model
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from utils.util_val.val_offical import LaneEval
from models.model.single_camera_bev import *


model_path = '/dataset/model/apollo//0516/ep049.pth' #model path of verification

''' parameter from config '''
config_file = './apollo_config.py'
configs = load_config_module(config_file)
test_json_paths = configs.test_json_paths
x_range = configs.x_range
y_range = configs.y_range
meter_per_pixel = configs.meter_per_pixel


'''Post-processing parameters '''
post_conf = 0.9 # Minimum confidence on the segmentation map for clustering
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 15 # The minimum number of points in a cluster
tmp_save_path = '/workspace/tmp_apollo' #tmp path for save intermediate result

class PostProcessDataset(Dataset):
    def __init__(self, model_res_save_path, postprocess_save_path,test_json_paths):
        self.valid_data = os.listdir(model_res_save_path)
        self.postprocess_save_path = postprocess_save_path
        self.model_res_save_path = model_res_save_path
        self.x_range = x_range
        self.meter_per_pixel = meter_per_pixel
        d_gt_res = {}
        with open(test_json_paths, 'r') as f:
            for i in f.readlines():
                line_content = json.loads(i.strip())
                # print('hah') #'raw_file', 'cam_height', 'cam_pitch', 'centerLines', 'laneLines', 'centerLines_visibility', 'laneLines_visibility'
                lanes = []
                for lane_idx in range(len(line_content['laneLines'])):
                    lane_selected = np.array(line_content['laneLines'][lane_idx])[
                        np.array(line_content['laneLines_visibility'][lane_idx]) > 0.5]
                    lanes.append(lane_selected.tolist())
                d_gt_res[line_content['raw_file']] = lanes
        self.d_gt_res = d_gt_res

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, item):
        loaded = np.load(os.path.join(self.model_res_save_path, self.valid_data[item]))
        prediction = (loaded[:, 0:1, :, :], loaded[:, 1:3, :, :])
        offset_y = loaded[:, 3:4, :, :][0][0]
        z_pred = loaded[:, 4:5, :, :][0][0]
        files = self.valid_data[item].split('.')[0].split('__')
        canvas, ids = embedding_post(prediction, post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
        lines = bev_instance2points_with_offset_z(canvas, max_x=self.x_range[1],
                                    meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),offset_y=offset_y,Z=z_pred)
        frame_lanes_pred = []
        for lane in lines:
            pred_in_persformer = np.array([-1*lane[1],lane[0],lane[2]])
            y = np.linspace(min(pred_in_persformer[1]),max(pred_in_persformer[1]),40)
            f_x = np.polyfit(pred_in_persformer[1],pred_in_persformer[0],3)
            f_z = np.polyfit(pred_in_persformer[1], pred_in_persformer[2], 3)
            pred_in_persformer = np.array([np.poly1d(f_x)(y),y,np.poly1d(f_z)(y)])
            frame_lanes_pred.append(pred_in_persformer.T.tolist())
        gt_key = 'images' + '/' + files[0] + '/' + files[1] + '.jpg'
        frame_lanes_gt = self.d_gt_res[gt_key]
        with open(os.path.join(self.postprocess_save_path, files[0]+'_'+files[1] + '.json'), 'w') as f1:
            json.dump([frame_lanes_pred, frame_lanes_gt], f1)
        return torch.zeros((3, 3))


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
    time1 = int(time.time()) 
    np_save_path = os.path.join(tmp_save_path, str(time1) + '_np')
    print("np_save_path:", np_save_path)
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
    postprocess = PostProcessDataset(np_save_path, res_save_path, test_json_paths)
    postprocess_loader = DataLoader(dataset=postprocess,
                                    batch_size=32,
                                    num_workers=16,
                                    shuffle=False)
    for item in tqdm(postprocess_loader):
        continue
    ''' verification by official tools in Gen-LaneNet'''
    lane_eval = LaneEval()
    res_list = os.listdir(res_save_path)
    for item in tqdm(res_list):
        with open(os.path.join(res_save_path, item), 'r') as f:
            res = json.load(f)
        lane_eval.bench_all(res[0], res[1])
    lane_eval.show()
    shutil.rmtree(np_save_path)
    shutil.rmtree(res_save_path)


if __name__ == '__main__':
    val()