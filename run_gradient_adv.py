import numpy as np
import argparse
import os
import re
import sys
from pathlib import Path
from art.attacks import evasion as eva
from adv_lib.utils import ExtractDtection, SavePlotWithPrediction, AdvImgCreator
from adv_lib import load_pytorch_pre_trained_model, ModifyPyTorchObjectDetector


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help='the name of a pre-trained pytorch model')
    parser.add_argument('--img_type', type=str, default='jpg',
                        help='optional, the image file type; default is jpg', required=False)
    parser.add_argument('--raw_image_root', type=str,
                        help='the folder that stores raw images and the annotation file')
    parser.add_argument('--adv_image_root', type=str,
                        help='the folder where adversarial images will be saved')
    parser.add_argument('--display_raw_pred_root',
                        help='optional, the foler where model prediction displays for raw images will be saved', required=False)
    parser.add_argument('--display_adv_pred_root',
                        help='optional, the foler where model prediction displays for adversarial images will be saved', required=False)
    parser.add_argument('--atk_eps', type=float, default=5,
                        help='optional, PGD attack eps paremeter; default is 5', required=False),
    parser.add_argument('--atk_eps_step', type=int, default=2,
                        help='optional, PGD attack eps_step paremeter, default is 2', required=False)
    parser.add_argument('--atk_max_iter', type=int, default=10,
                        help='optional, PGD attack max_iter parameter, default is 10', required=False)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    model_name = args.model_name
    target_model = load_pytorch_pre_trained_model(model_name, 91)
    target_model = ModifyPyTorchObjectDetector(
        model=target_model, clip_values=(0, 255))

    raw_img_names = [p for p in os.listdir(args.raw_image_root) if re.search(
        "%s$" % args.img_type, p) is not None]
    adv_img_names = ['adv_'+name for name in raw_img_names]

    pgd = eva.ProjectedGradientDescent(estimator=target_model,
                                       eps=args.atk_eps, eps_step=args.atk_eps_step, max_iter=args.atk_max_iter)
    adv_generator = AdvImgCreator(args.raw_image_root, args.adv_image_root)

    pred = ExtractDtection()
    box_fornt_size = (5, 6, 6)

    for raw, adv in zip(raw_img_names, adv_img_names):
        adv_generator.create_img(pgd, raw, adv)
        if args.display_raw_pred_root is not None:
            raw_pred_plotor = SavePlotWithPrediction(
                target_model, pred, args.display_raw_pred_root)
            raw_pred_plotor.display_model_prediction(
                adv_generator.img_arr, raw, box_fornt_size)
        if args.display_adv_pred_root is not None:
            adv_pred_plotor = SavePlotWithPrediction(
                target_model, pred, args.display_adv_pred_root)
            adv_pred_plotor.display_model_prediction(
                adv_generator.atk_img_arr, adv, box_fornt_size)
