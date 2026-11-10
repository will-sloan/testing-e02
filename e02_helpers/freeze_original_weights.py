def freeze_original_weights(qwen):
    for parameter in qwen.parameters():
        parameter.requires_grad = False
