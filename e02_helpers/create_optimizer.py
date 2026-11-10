def create_optimizer(parameters):
    import torch

    return torch.optim.AdamW(parameters, lr=0.0001)
