from typing import Dict
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


def load_checkpoint(checkpoint, net, optimizer=None, map_loc="cuda"):
    sd = torch.load(checkpoint, map_location=map_loc)
    net.load_state_dict(sd['models'])
    if optimizer and sd['optimizer']:
        optimizer.load_state_dict(sd['optimizer'])
    return sd


def resume_training(checkpoint: Dict,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    # Load checkpoint
    sd = load_checkpoint(checkpoint, model, optimizer)
    # TODO: Fix warning
    scheduler.step(sd['epoch'])


def load_model(model, model_state_file):
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'model_state' in pretrained_dict:
        pretrained_dict1 = pretrained_dict['model_state']
    elif 'state_dict' in pretrained_dict:
        pretrained_dict1 = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict1.items()
                       if k[6:] in model_dict.keys()}
    if len(pretrained_dict) == 0:
        pretrained_dict = {k: v for k, v in pretrained_dict1.items()
                           if k in model_dict.keys()}
    count = 0
    for k in model_dict.keys():
        if k not in pretrained_dict.keys():
            count += 1
            print(k)
    print(count)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
