import torch
import functools
from typing import Sequence

def baseline(ppls, config):
    """
    Baseline function that returns a tensor of the same shape as ppls,
    filled with the number of experts per token.
    """
    k = config.num_experts_per_tok
    return torch.full_like(ppls, k, dtype=torch.int64)

@functools.cache
def _get_cfg_boundaries(cfg: tuple[float], base_k: int) -> torch.Tensor:
    """
    Prepare the boundaries tensor for bucketize.
    The boundaries should be in ascending order for torch.bucketize.
    """
    if len(cfg) < base_k:
        cfg = cfg + (0.0,) * (base_k - len(cfg))
    boundaries = torch.tensor(
        cfg[::-1], 
        dtype=torch.float32
    )
    return boundaries

def _calc_segment(
    cfg: tuple[float],
    ppls: torch.FloatTensor,
    config
):
    base_k = config.num_experts_per_tok
    boundaries = _get_cfg_boundaries(cfg, base_k)
    return torch.bucketize(ppls, boundaries, right=True)

def spec_default(ppls, config):
    return _calc_segment((6, 1.17, 1.07, 1.07), ppls, config)

spec_default4 = spec_default

def spec_default3(ppls, config):
    return _calc_segment((6, 1.17, 1.07), ppls, config)

def spec_default2(ppls, config):
    return _calc_segment((6, 1.17), ppls, config)

def spec_default1(ppls, config):
    return _calc_segment((6,), ppls, config)

def spec_aggresive(ppls, config):
    return _calc_segment((6, 1.17, 1.07, 1.035, 1.005, 1.005), ppls, config)

def spec_layerwise1(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(6, 18) or lid in range(num_layers-22, num_layers):
            cfg = (x + 1.0 for x in base_spec_formula)
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_layerwise2(ppls, config):
    num_layers = config.num_hidden_layers
    spec_formula = (6, 1.17, 1.07, 1.035, 1.005, 1.005)
    all_layer_ks = ()
    for lid in range(num_layers):
        # if lid in range(0, num_layers, 4):
        # if lid in range(num_layers//2, num_layers):
        # if lid in range(num_layers-22, num_layers):
        # if lid in range(6, 18):
        if lid in range(6, 18) or lid in range(num_layers-14, num_layers):
            cfg = spec_formula # (x + 1.0 for x in spec_formula)
        else:
            cfg = spec_formula[:2]
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

# based on default4, seek for even better benefit
def spec_layerwise3(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(6, 18) or lid in range(num_layers-22, num_layers):
            cfg = (x + 1.0 for x in base_spec_formula)
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

# ablation cases for early layers
def spec_layerwise1_early0(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(0, 12) or lid in range(num_layers-22, num_layers):
            cfg = (x + 1.0 for x in base_spec_formula)
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

spec_layerwise_early0 = spec_layerwise1_early0

def spec_layerwise1_early1(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(1, 13) or lid in range(num_layers-22, num_layers):
            cfg = (x + 1.0 for x in base_spec_formula)
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

spec_layerwise_early1 = spec_layerwise1_early1

def spec_layerwise1_early4(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(4, 16) or lid in range(num_layers-22, num_layers):
            cfg = (x + 1.0 for x in base_spec_formula)
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

spec_layerwise_early4 = spec_layerwise1_early4

def spec_default1_mask2025(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(20, 25):
            # do not apply any dyn ks
            cfg = ()
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_default2_mask2025(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(20, 25):
            # do not apply any dyn ks
            cfg = ()
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_default3_mask2025(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(20, 25):
            # do not apply any dyn ks
            cfg = ()
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_default4_mask2025(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (6, 1.17, 1.07, 1.07)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(20, 25):
            # do not apply any dyn ks
            cfg = ()
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_opt1_mask2025(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = (10.0, 6.58, 1.275, 1.0)
    all_layer_ks = ()
    for lid in range(num_layers):
        if lid in range(20, 25):
            # do not apply any dyn ks
            cfg = ()
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

spec_with_list_full_layers = _calc_segment

def spec_with_list_layer_range(cfg: tuple[float], layer_range: tuple[int, ...], ppls, config):
    num_layers = config.num_hidden_layers
    basic = _calc_segment(tuple(cfg), ppls, config)
    all = basic.unsqueeze_(0).repeat(num_layers, 1)
    # Mask the layer range.
    all[slice(*layer_range)] = config.num_experts_per_tok

    return all

def constant_k(k: int, ppls, config):
    return torch.full_like(ppls, k, dtype=torch.int64)

def spec_from_layer_cfgs(
    layer_cfgs: Sequence[Sequence[float]],
    ppls: torch.FloatTensor,
    config,
):
    """
    layer_cfgs: length = num_layers
      each element is a t-space cfg tuple/list, e.g. (t0,t1,t2,t3)
    returns:
      all_k: int tensor (num_layers, num_tokens)
    """
    num_layers = int(config.num_hidden_layers)

    if len(layer_cfgs) != num_layers:
        raise ValueError(f"layer_cfgs length {len(layer_cfgs)} != num_layers {num_layers}")

    ks = []
    for cfg in layer_cfgs:
        ks.append(_calc_segment(tuple(float(x) for x in cfg), ppls, config))
    return torch.stack(ks, dim=0)

if __name__ == "__main__":
    print(spec_with_list_layer_range(
        (6, 1.17, 1.07, 1.07, 1.035, 1.005, 1.005),
        (5, 10, 2),
        torch.tensor([1.5, 2.0, 2.5, 6.2, 6.0, 5.9, 1.165, 1.06, 1.0]),
        type('Config', (), {'num_experts_per_tok':8, 'num_hidden_layers':12})
    ))
