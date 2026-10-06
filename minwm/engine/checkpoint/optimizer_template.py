"""Rebuild sparse optimizer load destinations from saved DCP metadata.

Unlike a fresh optimizer, metadata includes every saved Muon momentum tensor
and omits AdamW state for parameters that have never received gradients.
"""
import math
import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def optimizer_load_template(metadata, root_key, opt_key, current, model_state):
    prefix = (root_key, opt_key)
    paths = metadata.planner_data or {}
    if not any(tuple(path[:2]) == prefix for path in paths.values()):
        raise KeyError(f'Checkpoint has no optimizer {root_key}/{opt_key}')
    names = {name for group in current['param_groups'] for name in group['params']}
    states = {}
    allowed = {'step', 'use_muon', 'momentum_buffer', 'moment1', 'moment2',
               'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}
    for flat, path in paths.items():
        if tuple(path[:3]) != (*prefix, 'state'):
            continue
        if len(path) != 5:
            raise ValueError(f'Unsupported optimizer state path: {path}')
        _, _, _, name, field = path
        if name not in names or name not in model_state or field not in allowed:
            raise ValueError(f'Unknown optimizer state: {path}')
        meta = metadata.state_dict_metadata[flat]
        parameter = model_state[name]
        if isinstance(meta, TensorStorageMetadata):
            shape, dtype = tuple(meta.size), meta.properties.dtype
            if shape == () and field == 'step':
                value = torch.empty((), dtype=dtype, device='cpu')
            elif shape == tuple(parameter.shape):
                value = torch.empty_like(parameter, dtype=dtype)
            elif (field == 'momentum_buffer' and parameter.ndim > 2
                  and shape == (parameter.shape[0], math.prod(parameter.shape[1:]))):
                value = torch.empty_like(parameter, dtype=dtype).reshape(shape)
            else:
                raise ValueError(f'Unsupported optimizer tensor shape: {path} {shape}')
        else:
            value = current['state'].get(name, {}).get(field)
        states.setdefault(name, {})[field] = value
    return {**current, 'state': states}
