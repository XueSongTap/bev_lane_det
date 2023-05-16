import torch
from torch import Tensor, nn as nn
from torch.optim import AdamW


class PushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,1,h,w], float tensor
    gt: gt, [b,1,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(PushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape == gt.shape)
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # B, 1, H, W
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b]
            instance_means = {}
            for i in range(1, C + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue

                instance_mean = bfeat[instance_mask].mean()
                instance_means[i] = instance_mean
                instance_loss = torch.clamp(torch.abs(bfeat[instance_mask] - instance_mean) - self.margin_var,
                                           min=0.0) ** 2
                pull_loss.append(instance_loss.mean())
            for i in range(1, C + 1):
                for j in range(1, C + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_means or j not in instance_means:
                        continue
                    instance_loss = torch.clamp(2 * self.margin_dist - torch.abs(instance_means[i] - instance_means[j]),
                                               min=0.0) ** 2
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss


def rank_print(str):
    rank = torch.distributed.get_rank()
    print(str, " @ rank {}".format(rank))


class NDPushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,N,h,w], float tensor
    gt: gt, [b,N,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(NDPushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape[2:] == gt.shape[2:])
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # [B, N, H, W] = fm, [B, 1, H, W]  = gt
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b][0]
            instance_centers = {}
            for i in range(1, int(C) + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue
                pos_featmap = bfeat[:, instance_mask].T.contiguous()  #  mask_num x N
                instance_center = pos_featmap.mean(dim=0, keepdim=True)  # N x mask_num (mean)-> N x 1
                instance_centers[i] = instance_center
                # TODO xxx
                instance_loss = torch.clamp(torch.cdist(pos_featmap, instance_center) - self.margin_var, min=0.0)
                pull_loss.append(instance_loss.mean())
            for i in range(1, int(C) + 1):
                for j in range(1, int(C) + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_centers or j not in instance_centers:
                        continue
                    instance_loss = torch.clamp(
                        2 * self.margin_dist - torch.cdist(instance_centers[i], instance_centers[j]), min=0.0)
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss


class MSPushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [[b,1,h,w], ... ], list of float tensor for multi-scale
    gt: gt, [[b,1,h,w], ... ], list of long tensor for multi-scale, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(MSPushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmaps, gts):
        # rank_print("FORWARD")
        assert len(featmaps) == len(gts)
        for fm, gt in zip(featmaps, gts):
            assert (fm.shape == gt.shape)

        pull_loss = []
        push_loss = []

        batch_size = featmaps[0].shape[0]
        C = gts[0][gts[0] < self.ignore_label].max().item()
        # B, 1, H, W

        # BS
        for b in range(batch_size):
            bfeats = [fm[b] for fm in featmaps]
            bgts = [gt[b] for gt in gts]
            instance_means = {}

            # Instance
            for i in range(1, C + 1):
                # rank_print("instance {}".format(i))
                instance_masks = [bgt == i for bgt in bgts]
                scales_instance = []
                for bfeat, mask in zip(bfeats, instance_masks):
                    if mask.sum() == 0:
                        continue
                    single_scale_instance = bfeat[mask]
                    scales_instance.append(single_scale_instance)

                if len(scales_instance) == 0:
                    continue

                instance_mean = torch.cat(scales_instance).mean()
                instance_means[i] = instance_mean

                scale_instance_loss = []
                for bfeat, mask in zip(bfeats, instance_masks):
                    if mask.sum() == 0:
                        continue
                    scale_instance_loss.append(
                        (torch.clamp(
                            torch.abs(bfeat[mask] - instance_mean) - self.margin_var,
                            min=0.0) ** 2).mean())
                instance_loss = sum(scale_instance_loss)
                pull_loss.append(instance_loss)
            # rank_print("pull_loss DONE @ {}".format(b))
            for i in range(1, C + 1):
                for j in range(1, C + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_means or j not in instance_means:
                        continue
                    instance_loss = torch.clamp(
                        2 * self.margin_dist - torch.abs(instance_means[i] - instance_means[j]),
                        min=0.0) ** 2
                    push_loss.append(instance_loss)
            # rank_print("push_loss DONE @ {}".format(b))
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * fake_loss(featmaps)

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * fake_loss(featmaps)  # Fake loss
        return push_loss + pull_loss + fake_loss(featmaps) * 0.0


# A naive tensor.mean() * 0.0 loss to make grad graph stable.
def fake_loss(pred):
    loss = 0
    if isinstance(pred, dict):
        for k, v in pred.items():
            loss += fake_loss(v)
    elif isinstance(pred, list):
        for i in pred:
            loss += fake_loss(i)
    elif isinstance(pred, Tensor):
        loss += pred.mean()
    elif isinstance(pred, tuple):
        for i in pred:
            loss += fake_loss(i)
    else:
        print("fake loss {}".format(type(pred)))
        raise NotImplementedError()

    return loss * 0.0


class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den


if __name__ == '__main__':
    import torchvision as tv
    ND = 32
    model = tv.models.resnet18(False)
    model = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.Conv2d(512, ND, 1, 1)
    )

    model.train()
    gt = torch.zeros(4, 1, 4, 4, dtype=torch.long)
    gt[:, :, :2, :1] = 1
    gt[:, :, :2, 1:2] = 7
    gt[:, :, :2, 2:3] = 6
    gt[:, :, :2, 3:] = 2
    gt[:, :, 2:, :2] = 4
    gt[:, :, 2:, 2:] = 5
    print(gt)
    l = NDPushPullLoss(1.0, 1., 1., 3., 100)
    optim = AdamW(model.parameters())
    for i in range(51):
        data = torch.rand(4, 3, 128, 128)
        optim.zero_grad()
        ret = model(data)
        # print(ret.shape)
        # exit()
        # print(ret)
        # print(ret.shape)
        loss = l(ret, gt)
        loss.backward()
        optim.step()
        print(loss.item())
        if loss < 0.1:
            break
        # if i % 10 == 0:
        #     print(ret[:, : ])

    mat = ret.detach().numpy()
    print(mat[0].shape)

    import numpy as np
    from sklearn.cluster import DBSCAN

    for i in range(4):
        clustering = DBSCAN(eps=3., min_samples=2, metric='l2').fit(mat[i].reshape(ND, 16).T)
        print(mat[i])
        print("GT:\n", gt[i])
        print(np.array(clustering.labels_).reshape(4, 4), clustering)
