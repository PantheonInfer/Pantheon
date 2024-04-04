import torch
from collections import OrderedDict


def load_state_dict(path):
    ckpt = torch.load(path)
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[1:])
        new_state_dict[name] = v
    return new_state_dict