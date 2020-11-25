import dgl
import torch
import torch.nn as nn
from torchsummary import summary

class GNN(nn.Module):
    """
    Creates a GNN meta-model based on the inner model construct
    """
    def __init__(self):
        # create GNN modeled after NN in inner model