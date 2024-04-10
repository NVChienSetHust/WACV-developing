import pickletools

import torch
import ast, pickle
from fickling.fickle import Pickled

import pprint

import pickle
from classification.models.mcunet.tinynas.nn.networks import ProxylessNASNets

from torchvision.models import (
    resnet18,
    resnet50,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
)
import torch.nn as nn
from core.utils import dist
import wandb
import os
# building mcu models
from quantize.custom_quantized_format import build_quantized_network_from_cfg
from quantize.quantize_helper import create_scaled_head, create_quantized_head

from core.utils.config import (
    config,
    load_transfer_config,
    update_config_from_wandb,
)

# import for mcunet training
from tqdm import tqdm
import json

from classification.models.mcunet.model_zoo import build_model
from classification.models.mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms

def build_config(config_file_path):
    load_transfer_config(config_file_path)

    # Init wandb
    if dist.rank() <= 0:
        print("Initialize wandb run")
        wandb.init(project=config.project_name, config=config)
        os.makedirs(os.path.join("./scratch", "checkpoints", wandb.run.id))

    if config.wandb_sweep:
        update_config_from_wandb(wandb.config)
        wandb.config.update(config, allow_val_change=True)


def build_mcu_model():
    cfg_path = f"assets/mcu_models/{config.net_config.net_name}.pkl"
    cfg = torch.load(cfg_path)
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    if config.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif config.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    return model

config_file_path = f"transfer.yaml"
build_config(config_file_path)
# Path to your pickle file
pickle_file_path = f'{config.net_config.net_name}.pkl'

# model = build_mcu_model()

# model, total_neurons = get_model()
model, resolution, description = build_model(config.net_config.net_name, pretrained=True)
total_neurons = 0

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        total_neurons += m.weight.shape[0]
print("Model attributes", model)
print("Model: ", config.net_config.net_name)
print("image_size: ", config.data_provider.image_size)
print("Total neuron: ", total_neurons)

# Open a file for writing
output_file_path = f"{config.net_config.net_name}_architecture.txt"
with open(output_file_path, "w") as f:
    # Pretty-print the loaded configuration to the file
    pprint.pprint(model, stream=f)

print(f"Configuration printed to {output_file_path}")


# # Open the pickle file in binary read mode
# with open(pickle_file_path, 'rb') as f:
#     # Read the contents of the pickle file
#     pickle_data = pickle.load(f)

# fickled_object = Pickled.load(pickle_data)
# print(ast.dump(fickled_object.ast))


# # Use pickletools to disassemble and print the contents of the pickle file
# try:
#     pickletools.dis(pickle_data)
# except Exception as e:
#     print("Failed to analyze pickle data:", e)
