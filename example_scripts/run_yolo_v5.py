import os
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from adv_lib import CoCoTransforms, compute_detection_metircs
from adv_lib.predict_method import yolo_predict_method


if __name__ == "__main__":

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    coco_img_root = './sample_COCO_val_2017_data'

    img_data = CocoDetection(root=coco_img_root, 
                         annFile=os.path.join(coco_img_root, 'annotations.json'), 
                         transforms=CoCoTransforms('xywh', 'xyxy',[]))

    dataloader = DataLoader(
            img_data,
            batch_size=5,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: batch
        )
    
    out = yolo_predict_method(yolo, dataloader)
    print(compute_detection_metircs(out, dataloader))