import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from models import MLPMixer, MixerBlock

import torch_pruning as tp

model = MLPMixer()
example_inputs = torch.randn(1, 1, 28, 28)

# 0. importance criterion for parameter selections
imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

# 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

        
# 2. Pruner initialization
iterative_steps = 5 # You can prune your model to the target pruning ratio iteratively.
pruner = tp.pruner.MagnitudePruner(
    model, 
    example_inputs, 
    global_pruning=True, # If False, a uniform ratio will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target ratio
    pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    # 3. the pruner.step will remove some channels from the model with least importance
    pruner.step()
    
    # 4. Do whatever you like here, such as fintuning
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print(model(example_inputs).shape)
    print(
        "  Iter %d/%d, Params: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
    )
    print(
        "  Iter %d/%d, MACs: %.2f G => %.2f G"
        % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
    )
    # finetune your model here
    # finetune(model)
    # ...