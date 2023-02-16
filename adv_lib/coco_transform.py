import torch
from PIL.Image import Image
from torchvision import transforms as T
from typing import Tuple, List, Dict, Callable
from torchvision.ops.boxes import box_convert
from adv_lib.utils import device

default_transforms = [T.PILToTensor(), T.ConvertImageDtype(torch.float)]

class CoCoTransforms:

    def __init__(self, input_box_format: str, model_box_format: str, img_transforms:List[Callable]) ->None:

        self.input_box_format = input_box_format
        self.model_box_format = model_box_format
        self.img_transforms = img_transforms

    def img_transformer(self, image: Image) -> torch.Tensor:

        transforms = []
        if not isinstance(image, Image):
            transforms.append(T.ToPILImage())
        transforms += self.img_transforms
        return T.Compose(transforms)(image)

    def target_transformer(self, target: List[Dict]) -> List[Dict]:

        boxes = torch.as_tensor([obj['bbox']
                                for obj in target], dtype=torch.float32)
        labels = torch.tensor([obj['category_id'] for obj in target])
        tf_target = {}
        tf_target['boxes'] = box_convert(
            boxes, self.input_box_format, self.model_box_format).to(device)
        tf_target['labels'] = labels.to(device)
        return tf_target

    def __call__(self, image: Image, target: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        img = self.img_transformer(image)
        if isinstance(img, torch.Tensor):
            img = img.to(device)
        target_tensor = self.target_transformer(target)

        return img, target_tensor