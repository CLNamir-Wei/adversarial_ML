import argparse
import os
import re
from art.attacks import evasion as eva
from adv_lib.utils import ExtractDtection, SavePlotWithPrediction, AdvImgCreator
from adv_lib import load_pytorch_pre_trained_model, ModifyPyTorchObjectDetector


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model', type=str,
                        help='the name of a pre-trained pytorch model for the white box attack')
    parser.add_argument('--victim_model', type=str,
                        help='the name of a pre-trained pytorch model for the transfer attack', required=False)
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

    source_model = load_pytorch_pre_trained_model(args.source_model, 91)
    source_model = ModifyPyTorchObjectDetector(
        model=source_model, clip_values=(0, 255))

    if args.victim_model is None:
        victim_model = source_model
    else:
        victim_model = load_pytorch_pre_trained_model(args.victim_model, 91)
        victim_model = ModifyPyTorchObjectDetector(model=victim_model, clip_values=(0,255))

    raw_img_names = [p for p in os.listdir(args.raw_image_root) if re.search(
        "%s$" % args.img_type, p) is not None]
    adv_img_names = ['adv_'+name for name in raw_img_names]

    pgd = eva.ProjectedGradientDescent(estimator=source_model,
                                       eps=args.atk_eps, eps_step=args.atk_eps_step, max_iter=args.atk_max_iter)
    adv_generator = AdvImgCreator(args.raw_image_root, args.adv_image_root)

    pred = ExtractDtection()
    box_fornt_size = (5, 6, 6)

    for raw, adv in zip(raw_img_names, adv_img_names):
        adv_generator.create_img(pgd, raw, adv)
        if args.display_raw_pred_root is not None:
            raw_pred_plotor = SavePlotWithPrediction(
                victim_model, pred, args.display_raw_pred_root)
            raw_pred_plotor.display_model_prediction(
                adv_generator.img_arr, raw, box_fornt_size)
        if args.display_adv_pred_root is not None:
            adv_pred_plotor = SavePlotWithPrediction(
                victim_model, pred, args.display_adv_pred_root)
            adv_pred_plotor.display_model_prediction(
                adv_generator.atk_img_arr, adv, box_fornt_size)
