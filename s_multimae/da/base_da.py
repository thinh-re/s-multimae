import abc
from typing import List, Optional, Tuple
from torch import nn, Tensor
from PIL import Image


class BaseDataAugmentation(nn.Module):
    def __init__(self):
        super(BaseDataAugmentation, self).__init__()

    @abc.abstractmethod
    def forward(
        self,
        image: Image.Image,
        depth: Image.Image,
        gt: Optional[Image.Image] = None,
        ranking_gt: Optional[Image.Image] = None,
        multi_gts: Optional[List[Image.Image]] = None,
        is_transform: bool = True,  # is augmented?
        is_debug: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Usual case:
            If gt is provided, return [image, depth, gt]
            Otherwise, return [image, depth]

        When ranking_gt is provided, gt will be ignored
            Return [image, depth, ranking_gt]

        For debugging:
            Return [image, depth, gt|ranking_gt, unnormalized, Optional[ranking_gts]]
        """
        pass
