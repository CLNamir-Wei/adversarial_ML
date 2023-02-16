import itertools
import torch
from torch.utils.data import DataLoader
from typing import Callable, List, Dict
from adv_lib.utils import coco_label_translator
from adv_lib.coco_labels import COCO_INSTANCE_CATEGORY_NAMES, coco80


def torch_predict_method(torch_model: Callable, img_data_loader: DataLoader) -> List[Dict]:
    
    torch_model.eval()
    outputs = []
    for data_pair in img_data_loader:
        with torch.no_grad():
            outputs += [torch_model([x for x, _ in data_pair])]

    return list(itertools.chain.from_iterable(outputs))


def extract_yolo_pred(pred_tensor) -> Dict:
  
  y = pred_tensor[0]
  translator = lambda x: coco_label_translator(int(x), coco80, COCO_INSTANCE_CATEGORY_NAMES)
  boxes = y[:,:4]
  scores = y[:,4]
  labels = torch.tensor([translator(int_label) for int_label in y[:,5]])

  return {'boxes':boxes, 'labels':labels, 'scores':scores}
  
  
def yolo_predict_method(model: Callable, img_data_loader: DataLoader) -> List[Dict]:
    
    model.eval()
    outputs = []
    for data_pair in img_data_loader:
        with torch.no_grad():
            outputs += [extract_yolo_pred(model(x).xyxy) for x, _ in data_pair]

    return outputs