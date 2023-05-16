import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.openlane_data import OpenLane_dataset_with_offset,OpenLane_dataset_with_offset_val
from models.model.single_camera_bev import BEV_LaneDet

''' data split '''
train_gt_paths = '/dataset/openlane/lane3d_1000/training'
train_image_paths = '/dataset/openlane/images/training'
val_gt_paths = '/dataset/openlane/lane3d_1000/validation'
val_image_paths = '/dataset/openlane/images/validation'

model_save_path = "/dataset/model/openlane"

input_shape = (576,1024)
output_2d_shape = (144,256)

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=64,
    num_workers=12,
    shuffle=True
)

''' virtual camera config '''
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = np.array([[2081.5212033927246, 0.0, 934.7111248349433],
                                    [0.0, 2081.5212033927246, 646.3389987785433],
                                    [0.0, 0.0, 1.0]])
vc_config['vc_extrinsics'] = np.array(
        [[-0.002122161262459438, 0.010697496358766389, 0.9999405282331697, 1.5441039498273286],
            [-0.9999378331046326, -0.010968621415360667, -0.0020048117763292747, -0.023774034344867204],
            [0.010946522625388108, -0.9998826195688676, 0.01072010851209982, 2.1157397903843567],
            [0.0, 0.0, 0.0, 1.0]])
vc_config['vc_image_shape'] = (1920, 1280)


''' model '''
def model():
    return BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape,train=True)


''' optimizer '''
epochs = 50
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR



def train_dataset():
    train_trans = A.Compose([
                    A.Resize(height=input_shape[0], width=input_shape[1]),
                    A.MotionBlur(p=0.2),
                    A.RandomBrightnessContrast(),
                    A.ColorJitter(p=0.1),
                    A.Normalize(),
                    ToTensorV2()
                    ])
    train_data = OpenLane_dataset_with_offset(train_image_paths, train_gt_paths, 
                                              x_range, y_range, meter_per_pixel, 
                                              train_trans, output_2d_shape, vc_config)

    return train_data


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2()])
    val_data = OpenLane_dataset_with_offset_val(val_image_paths,val_gt_paths,
                                                trans_image,vc_config)
    return val_data



