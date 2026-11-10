def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
