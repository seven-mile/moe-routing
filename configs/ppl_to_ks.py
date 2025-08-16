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

def spec_default(ppls, config):
    return _calc_segment([6, 1.17, 1.07, 1.07], ppls, config)

spec_default4 = spec_default

def spec_default3(ppls, config):
    return _calc_segment([6, 1.17, 1.07], ppls, config)

def spec_default2(ppls, config):
    return _calc_segment([6, 1.17], ppls, config)

def spec_default1(ppls, config):
    return _calc_segment([6], ppls, config)

def spec_aggresive(ppls, config):
    return _calc_segment([6, 1.17, 1.07, 1.035, 1.005, 1.005], ppls, config)

def spec_layerwise1(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = [6, 1.17, 1.07]
    all_layer_ks = []
    for lid in range(num_layers):
        if lid in range(6, 18) or lid in range(num_layers-22, num_layers):
            cfg = [x + 1.0 for x in base_spec_formula]
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_layerwise2(ppls, config):
    num_layers = config.num_hidden_layers
    spec_formula = [6, 1.17, 1.07, 1.035, 1.005, 1.005]
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

# based on default4, seek for even better benefit
def spec_layerwise3(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = [6, 1.17, 1.07, 1.07]
    all_layer_ks = []
    for lid in range(num_layers):
        if lid in range(6, 18) or lid in range(num_layers-22, num_layers):
            cfg = [x + 1.0 for x in base_spec_formula]
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

# ablation cases for early layers
def spec_layerwise_early0(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = [6, 1.17, 1.07]
    all_layer_ks = []
    for lid in range(num_layers):
        if lid in range(0, 12) or lid in range(num_layers-22, num_layers):
            cfg = [x + 1.0 for x in base_spec_formula]
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)

def spec_layerwise_early1(ppls, config):
    num_layers = config.num_hidden_layers
    base_spec_formula = [6, 1.17, 1.07]
    all_layer_ks = []
    for lid in range(num_layers):
        if lid in range(0, 12) or lid in range(num_layers-22, num_layers):
            cfg = [x + 1.0 for x in base_spec_formula]
        else:
            cfg = base_spec_formula
        all_layer_ks.append(_calc_segment(cfg, ppls, config))

    return torch.stack(all_layer_ks)
