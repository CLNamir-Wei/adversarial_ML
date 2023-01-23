import os
import argparse
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from adv_lib import CoCoTransforms, torch_predict_method, load_pytorch_pre_trained_model, compute_detection_metircs


def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help='the name of a pre-trained pytorch model')
    parser.add_argument('--coco_img_root', type=str,
                        help='the folder that stores raw images and the annotation file')
    parser.add_argument('--data_box_formte', type=str, default='xywh',
                        help='the bounding box fomrate in the data set')
    parser.add_argument('--model_box_format', type=str, default='xyxy',
                        help='the bounding box formate used by the model')
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    my_model = load_pytorch_pre_trained_model(args.model_name, 91)

    img_data = CocoDetection(root=args.coco_img_root,
                             annFile=os.path.join(
                                 args.coco_img_root, 'annotations.json'),
                             transforms=CoCoTransforms(args.data_box_formte, args.model_box_format))

    dataloader = DataLoader(
        img_data,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: batch
    )

    out = torch_predict_method(my_model, dataloader)
    print(compute_detection_metircs(out, dataloader))
