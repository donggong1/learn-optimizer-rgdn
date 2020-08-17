import torch

def weights_init(m: torch.nn.Module):
    """init weights for a model

    :param m: a torch module
    :returns: 
    :rtype: 

    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
