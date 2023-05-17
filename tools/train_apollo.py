import sys
sys.path.append('/workspace/bev_lane_det')# 添加模块搜索路径
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR # 导入余弦退火学习率调度器
from torch.utils.data import DataLoader  # 导入数据加载器
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training # 导入加载和恢复模型的函数
from models.util.save_model import save_model_dp # 导入保存模型的函数
from models.loss import IoULoss, NDPushPullLoss  # 导入自定义的损失函数
from utils.config_util import load_config_module # 导入加载配置文件的函数
from sklearn.metrics import f1_score # 导入F1分数计算函数
import numpy as np

# 定义一个继承自nn.Module的类，将模型和损失函数组合在一起
class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0])) # 定义二元交叉熵损失函数
        self.iou_loss = IoULoss() # 定义IoU损失函数 
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200) # 定义自定义的NDPushPull损失函数
        self.mse_loss = nn.MSELoss() # 定义均方误差损失函数
        self.bce_loss = nn.BCELoss() # 定义二元交叉熵损失函数
        # self.sigmoid = nn.Sigmoid()
    # 正向传播函数
    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, train=True):
        res = self.model(inputs) # 调用模型进行预测
        pred, emb, offset_y, z = res[0] # 获取预测结果
        pred_2d, emb_2d = res[1]
        if train:
            ## 3d
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)  # 计算BEV分割损失和IoU损失
            loss_emb = self.poopoo(emb, gt_instance) # 计算嵌入向量损失
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y) # 计算偏移量损失
            loss_z = self.mse_loss(gt_seg * z, gt_z) # 计算高度损失
            loss_total = 3 * loss_seg + 0.5 * loss_emb # 计算总损失
            loss_total = loss_total.unsqueeze(0) # 将总损失转换成一维张量
            loss_offset = 60 * loss_offset.unsqueeze(0) # 将偏移量损失转换成一维张量并乘以60
            loss_z = 30 * loss_z.unsqueeze(0) # 将高度损失转换成一维张量并乘以30
            ## 2d
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)  # 计算2D分割损失和IoU损失
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance) # 计算2D嵌入向量损失
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d  # 计算2D总损失
            loss_total_2d = loss_total_2d.unsqueeze(0)  # 将2D总损失转换成一维张量
            return pred, loss_total, loss_total_2d, loss_offset, loss_z  # 返回预测结果和损失
        else:
            return pred # 返回预测结果

# 训练一个epoch的函数
def train_epoch(model, dataset, optimizer, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()  # 将模型设置为训练模式
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance) in enumerate(
            dataset):
        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()  # 将输入数据转移到GPU上
        gt_seg_data = gt_seg_data.cuda()  # 将BEV分割标签转移到GPU上
        gt_emb_data = gt_emb_data.cuda()  # 将嵌入向量标签转移到GPU上
        offset_y_data = offset_y_data.cuda()  # 将偏移量标签转移到GPU上
        z_data = z_data.cuda() # 将高度标签转移到GPU上
        image_gt_segment = image_gt_segment.cuda() # 将2D分割标签转移到GPU上
        image_gt_instance = image_gt_instance.cuda() # 将2D嵌入向量标签转移到GPU上
        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z = model(input_data,
                                                                                gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance) # 正向传播
        loss_back_bev = loss_total_bev.mean()  # 计算BEV总损失的平均值
        loss_back_2d = loss_total_2d.mean() # 计算2D总损失的平均值
        loss_offset = loss_offset.mean() # 计算偏移量损失的平均值
        loss_z = loss_z.mean() # 计算高度损失的平均值
        loss_back_total = loss_back_bev + 0.5 * loss_back_2d + loss_offset + loss_z  # 计算总损失
        ''' caclute loss '''
        optimizer.zero_grad() # 清空梯度
        loss_back_total.backward()  # 反向传播计算梯度
        optimizer.step() # 更新模型参数
        if idx % 50 == 0:
            print(idx, loss_back_bev.item(), '*' * 10)
        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel() # 将BEV分割标签从GPU中取出并展平为一维数组
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel() # 将预测结果从GPU中取出并展平为一维数组
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)  # 计算F1分数
            loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(), 'z loss': loss_z.item(),
                            "F1_BEV_seg": f1_bev_seg} # 计算各项损失和F1分数
            # losses_show = loss_iter
            print(idx, loss_iter)

# worker_fuction  加载配置文件
def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)  # 加载配置文件

    ''' models and optimizer '''
    model = configs.model()  # 加载模型
    model = Combine_Model_and_Loss(model) # 将模型和损失函数组合在一起
    if torch.cuda.is_available(): 
        model = model.cuda() # 将模型转移到GPU上
    model = torch.nn.DataParallel(model)  # 将模型并行化处理
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)  # 定义优化器
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)  # 定义学习率调度器
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler) # 恢复模型和优化器
        else:
            load_checkpoint(checkpoint_path, model.module, None) # 仅恢复模型s

    ''' dataset '''
    Dataset = getattr(configs, "train_dataset", None)  # 获取数据集
    # 用于确认是否载入Dataset
    print(len(Dataset()))
    print("configs:", configs)
    if Dataset is None:
        Dataset = configs.training_dataset # 如果没有指定数据集，则使用默认数据集
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True) # 加载数据

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(configs.epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, configs, epoch)
        scheduler.step() # 更新学习率
        save_model_dp(model, optimizer, configs.model_save_path, 'ep%03d.pth' % epoch)# 保存模型
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./tools/apollo_config.py', gpu_id=[0,1])  # 调用worker_function函数，传入配置文件路径和GPU编号
