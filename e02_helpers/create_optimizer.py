def create_optimizer(parameters, *, learning_rate):
    import torch

    return torch.optim.AdamW(parameters, lr=learning_rate)
