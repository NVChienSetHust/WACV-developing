import torch
import numpy as np

from ..utils import dist
from ..utils.config import config

from .sgd_scale import SGDScale


REGISTERED_OPTIMIZER_DICT = {
    'sgd': (torch.optim.SGD, {'momentum': 0.9, 'nesterov': False}),
    'sgd_nomom': (torch.optim.SGD, {'momentum': 0, 'nesterov': False}),

    'sgd_scale': (SGDScale, {'momentum': 0.9, 'nesterov': False}),
    'sgd_scale_nomom': (SGDScale, {'momentum': 0., 'nesterov': False}),    
    
    'adam': (torch.optim.Adam, {}),
    'adamw': (torch.optim.AdamW, {}),
}


def default_wd_rules(model):
    net_params_with_wd = []
    net_params_without_wd = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if config.run_config.no_wd_keys is not None \
                    and np.any([key in name for key in config.run_config.no_wd_keys]):
                net_params_without_wd.append(param)
            else:
                net_params_with_wd.append(param)
    net_params = [
        {'params': net_params_with_wd, 'weight_decay': config.run_config.weight_decay},
        {'params': net_params_without_wd, 'weight_decay': 0},
    ]
    print(' * weight decay plan:')
    for p in net_params:
        print(p['weight_decay'], len(p['params']))
    return net_params


def build_optimizer(model):
    if hasattr(model, "module"):
        model = model.module
    if config.run_config.bs256_lr is not None:
        org_base_lr = config.run_config.base_lr
        config.run_config.base_lr = config.run_config.bs256_lr / 256 * config.data_provider.base_batch_size
        print(f' * Getting bs256_lr... Overwrite lr from {org_base_lr} to {config.run_config.base_lr} '
              f'(bs{config.data_provider.base_batch_size}), '
              f'total lr {config.run_config.base_lr * dist.size()} '
              f'(total bs {config.data_provider.base_batch_size * dist.size()})')
    if config.run_config.bias_only:
        param2update = []
        for name, p in model.named_parameters():
            if 'bias' in name:
                param2update.append(p)
        param2update.append(model[-2].weight)  # weight for the fc layer

        print('total param to update', len(param2update))
        net_params = param2update
    elif config.run_config.fc_only:
        from quantize.quantized_ops_diff import ScaledLinear, QuantizedConv2dDiff
        assert isinstance(model[-2], (ScaledLinear, QuantizedConv2dDiff)), type(model[-2])
        param2update = [model[-2].weight, model[-2].bias]
        print('total param to update', len(param2update))
        net_params = param2update
    elif config.run_config.n_block_update > 0:
        param2update = list(model.classifier.parameters())
        for blk in model.blocks[-config.run_config.n_block_update:]:
            param2update.extend(list(blk.parameters()))
        print('total param to update', len(param2update))
        net_params = param2update
    elif hasattr(model, 'wd_rules'):
        net_params = model.wd_rules(model)
    else:
        net_params = default_wd_rules(model)

    optimizer_class, default_params = REGISTERED_OPTIMIZER_DICT[config.run_config.optimizer_name]

    if config.run_config.get('optimizer_params', None) is not None:
        default_params.update(config.run_config.optimizer_params)

    default_params['lr'] = config.run_config.base_lr * dist.size()

    optimizer = optimizer_class(net_params, **default_params)
    return optimizer
