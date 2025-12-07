"""Transform combinators for object detection."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import random

from torch import Tensor
from PIL import Image


class RandomSelect:
    """Randomly select between two transforms.
    
    Args:
        transform1: First transform.
        transform2: Second transform.
        p: Probability of selecting transform1.
    """
    
    def __init__(self, transform1: Callable, transform2: Callable, p: float = 0.5) -> None:
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() < self.p:
            return self.transform1(image, target)
        return self.transform2(image, target)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class OneOf:
    """Apply one randomly chosen transform from a list.
    
    Args:
        transforms: List of transforms to choose from.
        p: Probability of applying any transform.
    """
    
    def __init__(self, transforms: List[Callable], p: float = 1.0) -> None:
        self.transforms = transforms
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        return random.choice(self.transforms)(image, target)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.transforms)} transforms, p={self.p})"
