import torch
from PIL.Image import Image
from torchvision import transforms as T
from typing import Union, Callable, Tuple, List, Any, Dict
from torchvision.ops.boxes import box_convert
from adv_lib.utils import device


class CoCoTransforms:

    def __init__(self, input_box_format: str, model_box_format: str):

        self.input_box_format = input_box_format
        self.model_box_format = model_box_format

    def img_transformer(self, image: Image):

        transforms = []
        if not isinstance(image, Image):
            transforms.append(T.ToPILImage())
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        return T.Compose(transforms)(image).to(device)

    def target_transformer(self, target: List[Dict]):

        boxes = torch.as_tensor([obj['bbox']
                                for obj in target], dtype=torch.float32)
        labels = torch.tensor([obj['category_id'] for obj in target])
        tf_target = {}
        tf_target['boxes'] = box_convert(
            boxes, self.input_box_format, self.model_box_format).to(device)
        tf_target['labels'] = labels.to(device)
        return tf_target

    def __call__(self, image: Image, target: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        img_tensor = self.img_transformer(image)
        target_tensor = self.target_transformer(target)

        return img_tensor, target_tensor
