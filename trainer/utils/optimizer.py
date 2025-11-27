import torch

def get_optimizer(cfg, parameters):
    if cfg.optimize.optimizer == "adam":
        optimize_cls = torch.optim.Adam
    elif cfg.optimize.optimizer == "adamw":
        optimize_cls = torch.optim.AdamW
    elif cfg.optimize.optimizer == "sgd":
        optimize_cls = torch.optim.SGD
    else:
        raise NotImplementedError("Not Implement {} optimizer".format(cfg.optimize.optimizer))
    
    return optimize_cls(
            parameters,
            lr=cfg.optimize.lr,
            weight_decay=cfg.optimize.get("weight_decay", 0.01),
            betas=cfg.optimize.get("betas", (0.9, 0.999)),
            eps=cfg.optimize.get("eps", 1e-08),
        )


def build_optimizer(cfg, model):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    norm_types = (
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
        LlamaRMSNorm
    )
    nodecay_names = []
    for name, mod in model.named_modules():
        if hasattr(mod, "bias"):
            nodecay_names.append("{}.bias".format(name))
        if isinstance(mod, norm_types):
            if hasattr(mod, "weight"):
                nodecay_names.append("{}.weight".format(name))
            if hasattr(mod, "scale"):
                nodecay_names.append("{}.scale".format(name))
    model_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay_params = [p for n, p in model_params if n not in nodecay_names]
    nodecay_params = [p for n, p in model_params if n in nodecay_names]
    grouped_parameters = [
        {
            "params": decay_params,
            "init_lr": cfg.optimize.lr,
            "lr": cfg.optimize.lr,
            "weight_decay": cfg.optimize.weight_decay,
        },
        {
            "params": nodecay_params,
            "init_lr": cfg.optimize.lr,
            "lr": cfg.optimize.lr,
            "weight_decay": 0.0,
        },
    ]
    num_all_params = sum(p.numel() for n, p in model_params)
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num all trained parameter tensors: {len(model_params)}, with {num_all_params:,} parameters")
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = get_optimizer(cfg, grouped_parameters)

    return optimizer
