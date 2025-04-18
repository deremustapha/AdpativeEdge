from torchprofile import profile_macs
import torch
import copy



def get_flops(net, data_shape):
    """
    Calculate the number of floating point operations (FLOPs) for a given neural network model.
    Args:
        net (torch.nn.Module): The neural network model for which to calculate FLOPs.
        data_shape (tuple): The shape of the input data as a tuple.
    Returns:
        int: The total number of multiply-accumulate operations (MACs) for the model.
    """
    device = net.parameters().__next__().device
    
    model = copy.deepcopy(net)
    #rm_bn_from_net(model)  # remove bn since it is eventually fused
    total_macs = profile_macs(model, torch.randn(*data_shape).to(device))
    del model
    return total_macs

def get_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params