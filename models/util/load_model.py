from typing import Dict
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
'''
这段代码是一个Python脚本，用于加载和恢复训练模型。

第一行 `from typing import Dict` 是导入了Python标准库中的 `Dict` 类型，用于定义一个字典类型的变量。

第二行 `import torch` 是导入PyTorch库，用于构建深度学习模型。

第三行 `from torch import nn` 是从PyTorch库中导入 `nn` 模块，用于定义神经网络模型。

第四行 `from torch.optim.lr_scheduler import _LRScheduler` 是从PyTorch库中导入 `_LRScheduler` 类型，用于定义学习率调度器。

第六行定义了一个函数 `load_checkpoint`，该函数用于加载模型检查点。

第七行 `sd = torch.load(checkpoint, map_location=map_loc)` 是使用PyTorch的 `load` 函数加载模型检查点，并将其保存到变量 `sd` 中。

第八行 `net.load_state_dict(sd['models'])` 是将模型参数加载到神经网络模型中。

第九行 `if optimizer and sd['optimizer']:` 是判断优化器和检查点中的优化器是否存在，如果存在，则将其加载到优化器中。

第十行 `return sd` 是返回模型检查点。

第十二行定义了一个函数 `resume_training`，该函数用于恢复训练。

第十三行 `print("Load checkpoint")` 是打印信息，提示正在加载模型检查点。

第十四行 `sd = load_checkpoint(checkpoint, model, optimizer)` 是调用 `load_checkpoint` 函数加载模型检查点，并将其保存到变量 `sd` 中。

第十六行 `scheduler.step(sd['epoch'])` 是将学习率调度器的步数设置为检查点中保存的训练轮数。

第十八行定义了一个函数 `load_model`，该函数用于加载预训练模型。

第十九行 `pretrained_dict = torch.load(model_state_file, map_location='cpu')` 是使用PyTorch的 `load` 函数加载模型参数，并将其保存到变量 `pretrained_dict` 中。

第二十行 `if 'model_state' in pretrained_dict:` 和第二十二行 `elif 'state_dict' in pretrained_dict:` 是判断模型参数是否保存在字典变量 `pretrained_dict` 中，并将其保存到变量 `pretrained_dict1` 中。

第二十一行 `pretrained_dict1 = pretrained_dict['model_state']` 是将模型参数保存到变量 `pretrained_dict1` 中。

第二十三行 `model_dict = model.state_dict()` 是获取神经网络模型的参数，并将其保存到变量 `model_dict` 中。

第二十四行 `pretrained_dict = {k[6:]: v for k, v in pretrained_dict1.items() if k[6:] in model_dict.keys()}` 是将预训练模型中的参数名称去掉前缀，并将其保存到变量 `pretrained_dict` 中。

第二十五行 `if len(pretrained_dict) == 0:` 是判断预训练模型中的参数是否与神经网络模型中的参数名称完全匹配，如果不匹配，则将预训练模型中的参数名称保持不变，并将其保存到变量 `pretrained_dict` 中。

第二十六行到第二十九行是计算神经网络模型中的参数名称与预训练模型中的参数名称不匹配的数量，并将其打印。

第三十行 `model_dict.update(pretrained_dict)` 是将预训练模型中的参数更新到神经网络模型中。

第三十一行 `model.load_state_dict(model_dict)` 是将更新后的神经网络模型参数加载到神经网络模型中。

第三十二行 `return model` 是返回神经网络模型。
'''

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
    print("Load checkpoint")
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
