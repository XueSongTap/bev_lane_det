from statistics import mean
import numpy as np
import torch

from torch import Tensor, multiprocessing as mp, distributed as dist
from torch.nn.parallel import DistributedDataParallel
from models.loss import fake_loss


def forward_on_cuda(gpu, gt_data, input_data, loss, model):
    input_data = naive_to_cuda(input_data, gpu)
    gt_data = naive_to_cuda(gt_data, gpu)
    prediction = model(input_data)
    if isinstance(model, DistributedDataParallel):
        loss_back, loss_iter = loss(prediction, gt_data, model.module)
    else:
        loss_back, loss_iter = loss(prediction, gt_data, model)
    return loss_back, loss_iter


def forward_on_cuda_with_fake_loss(task_name, gpu, gt_data, input_data, loss, model, with_fake_loss=True):
    input_data = naive_to_cuda(input_data, gpu)
    gt_data = naive_to_cuda(gt_data, gpu)
    prediction = model(input_data)
    if isinstance(model, DistributedDataParallel):
        loss_back, loss_iter = loss(prediction[task_name], gt_data, model.module)
    else:
        loss_back, loss_iter = loss(prediction[task_name], gt_data, model)

    # All fake loss to avoid find_unused_parameters
    if loss_back and with_fake_loss:
        loss_back += 0 * fake_loss(prediction)
    return loss_back, loss_iter


def get_task_val_data(task, repeat_self=False):
    if not task.val_iter:
        return None
    try:
        task_data = next(task.val_iter)
        return task_data
    except StopIteration:
        if repeat_self:
            task.val_iter = iter(task.val_dataset)
            task_data = next(task.val_iter)
            return task_data
        else:
            task.val_iter = None
            return None


def get_task_data(task, repeat_self=False):
    if not task.train_iter:
        return None
    try:
        task_data = next(task.train_iter)
        return task_data
    except StopIteration:
        if repeat_self:
            task.train_iter = iter(task.dataset)
            task_data = next(task.train_iter)
            return task_data
        else:
            task.train_iter = None
            return None


def train_one_task_with_fake_loss(model, task, loss_history, gpu):
    task_data = get_task_data(task, repeat_self=True)
    if task_data:
        task_input, task_gt = task_data
    else:
        return

    task_backward_loss, task_loss_log = forward_on_cuda_with_fake_loss(task.name, gpu, task_gt, task_input,
                                                                       task.loss, model)
    task_loss_log = dict([("{}.{}".format(task.name, k), v) for k, v in task_loss_log.items()])
    update_avg(loss_history, task_loss_log)
    return task_backward_loss, task_loss_log


def train_one_task(model, task, optimizer, gpu):
    if optimizer:
        optimizer.zero_grad()

    task_data = get_task_data(task, repeat_self=True)
    if task_data:
        task_input, task_gt = task_data
    else:
        # TODO seems never failed here, otherwise would crash.
        return None

    task_weight = task.weight
    task_backward_loss, task_loss_log = forward_on_cuda_with_fake_loss(
        task.name,
        gpu,
        task_gt,
        task_input,
        task.loss,
        model, with_fake_loss=True)
    weighted_loss = task_backward_loss * task_weight

    if optimizer:
        weighted_loss.backward()
        optimizer.step()
    task_loss_log = dict([("{}.{}".format(task.name, k), v) for k, v in task_loss_log.items()])
    return weighted_loss, task_loss_log


def naive_get_sample(data, index):
    if isinstance(data, Tensor):
        return data[index: index + 1, :, :, :]
    if isinstance(data, tuple):
        return [naive_get_sample(i, index) for i in data]
    if isinstance(data, list):
        return [naive_get_sample(i, index) for i in data]
    if isinstance(data, dict):
        return dict([(k, naive_get_sample(v, index)) for k, v in data.items()])


def naive_to_cuda(data, gpu):
    if isinstance(data, Tensor):
        return data.cuda(gpu, non_blocking=True)
    if isinstance(data, tuple):
        return [naive_to_cuda(i, gpu) for i in data]
    if isinstance(data, list):
        return [naive_to_cuda(i, gpu) for i in data]
    if isinstance(data, dict):
        return dict([(k, naive_to_cuda(v, gpu)) for k, v in data.items()])
    return data


def update_avg(loss_history, loss_iter):
    for k, v in loss_iter.items():
        if k in loss_history:
            loss_history[k].append(loss_iter[k])
        else:
            loss_history[k] = [loss_iter[k]]


def update_history(name, loss_history, loss_iter):
    for k, v in loss_iter.items():
        named_key = "{}.{}".format(name, k)
        if named_key in loss_history:
            loss_history[named_key].append(loss_iter[k])
        else:
            loss_history[named_key] = [loss_iter[k]]


def avg_losses(loss_history):
    return dict([(k, mean(v)) for k, v in loss_history.items()])


def mp_run(config, checkpoint_path, eval_only=False, worker=None):
    # TODO we always use this magic number. God bless.
    np.random.seed(3407)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, checkpoint_path, eval_only))


def mp_run_x(*args, **kwargs):
    # TODO we always use this magic number. God bless.
    np.random.seed(3407)
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    worker = kwargs['worker']
    del kwargs['worker']
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, *args, *kwargs))


def dist_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)
