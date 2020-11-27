import torch.nn.modules.module import Module
from torch import Tensor

class _DropnodesNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, drop_mask: Tensor) -> None:
        super(_DropnodesNd, self).__init__() #allow subclass objects to access these methods
        self.drop_mask = drop_mask

    def extra_repr(self) -> str:
        return 'drop mask={}'.format(self.drop_mask)

class Dropout2d(_DropoutNd):
    """Based on PyTorch Dropout Implementation
    Modified to dropout specific nodes
    """
    def forward(self, input: Tensor, drop_mask: Tensor) -> Tensor:
        