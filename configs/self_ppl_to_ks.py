import torch

def baseline(ppls, config):
    """
    Baseline function that returns a tensor of the same shape as ppls,
    filled with the number of experts per token.
    """
    k = config.num_experts_per_tok
    return torch.full_like(ppls, k, dtype=torch.int64)

def _calc_segment(
    cfg: list[float],
    ppls: torch.FloatTensor,
    config
):
    k = config.num_experts_per_tok
    ks = torch.full_like(ppls, k, dtype=torch.int64)
    for i, nk in zip(cfg, range(k)[::-1]):
        ks[ppls < i] = nk
    return ks

def self_default(ppls, config):
    return _calc_segment([2.0, 1.02, 1.004, 1.004], ppls, config)

self_default4 = self_default

def self_layerwise1(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = [2.0, 1.02, 1.004]
    all_layer_ks = []
    for lid in range(num_layers):
        if lid in range(6, 18) or lid in range(num_layers-22, num_layers):
            cfg = [x + 1.0 for x in base_spec_formula]
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def self_layerwise2(ppls, config):
    num_layers = config.num_hidden_layers
    spec_formula = [2.0, 1.02, 1.004, 1.004]
    all_layer_ks = []
    for lid in range(num_layers):
        # if lid in range(0, num_layers, 4):
        # if lid in range(num_layers//2, num_layers):
        # if lid in range(num_layers-22, num_layers):
        # if lid in range(6, 18):
        if lid in range(6, 18) or lid in range(num_layers-14, num_layers):
            cfg = spec_formula # [x + 1.0 for x in spec_formula]
        else:
            cfg = spec_formula[:2]
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)
