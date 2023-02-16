from importlib import import_module
from typing import Callable
from adv_lib.coco_transform import CoCoTransforms
from adv_lib.art_pytorch_obj_detecor import ModifyPyTorchObjectDetector
from adv_lib.utils import compute_detection_metircs

model_weigts = {
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "ssd300_vgg16": "SSD300_VGG16_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights"
}


def load_pytorch_pre_trained_model(model_name:str, n_class:int) -> Callable:

    model = getattr(import_module('torchvision.models.detection'), model_name)
    weights = getattr(import_module(
        'torchvision.models.detection'), model_weigts[model_name])
    return model(weights=weights.DEFAULT, num_classes=n_class)



