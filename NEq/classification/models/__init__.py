import torch
import json
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

from core.utils.config import config

# building mcu models
from quantize.custom_quantized_format import build_quantized_network_from_cfg
from quantize.quantize_helper import create_scaled_head, create_quantized_head


def build_mcu_model():
    cfg_path = f"NEq/assets/mcu_models/{config.net_config.net_name}.pkl"
    cfg = torch.load(cfg_path)
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    # print(model)

    if config.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif config.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    return model


def get_model():
    net_name = config.net_config.net_name
    print(f"Initialize model {net_name}")
    # MIT networks
    if "mcunet" in net_name or net_name == "proxyless-w0.3" or net_name == "mbv2-w0.35":
        model = build_mcu_model()
        total_neurons = 0

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                total_neurons += m.weight.shape[0]

        return model, total_neurons

    if net_name == "resnet18":
        model = resnet18()
    elif net_name == "resnet50":
        model = resnet50()
    elif net_name == "mbv2":
        model = mobilenet_v2()

    elif net_name == "pre_trained_resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif net_name == "pre_trained_resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif net_name == "pre_trained_mbv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    else:
        raise ValueError(f"No such model {config.net_config.net_name}")

    total_neurons = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total_neurons += m.weight.shape[0]

    return model, total_neurons
