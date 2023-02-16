import cv2
import torch
import numpy as np
import os
import shutil
import re
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import List, Dict, Callable, Tuple
from pathlib import Path
from adv_lib.coco_labels import COCO_INSTANCE_CATEGORY_NAMES, sysnonyms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def coco_label_translator(label_num:int, used_label_list:List, target_label_list:List) -> int:
    """
    Mapping the model output lables to a pre-defined label list, 
    such as PyTorch 91 COCO names at https://pytorch.org/vision/0.8/models.html.

    label_num: an int outputed by the model
    used_label_list: the COCO label list used by the model
    target_label_list: the target COCO label list, such as PyTorch 91 COCO names, used by ground truth data
    """
    text_label = used_label_list[label_num]
    if text_label in target_label_list: 
        return target_label_list.index(text_label)
    else:
        # assume that a sysnomym can only be associated to one key
        for key in sysnonyms:
            if text_label in sysnonyms[key]:
                return target_label_list.index(key)


def compute_detection_metircs(model_outputs: List[Dict], img_data_loader) -> Dict:

    metric = MeanAveragePrecision()
    targets = []
    for data_pair in img_data_loader:
        targets += [{k: v for k, v in y.items()} for _, y in data_pair]

    metric.update(model_outputs, targets)
    results = metric.compute()
    return {metric_name: value.item() for metric_name, value in results.items()
            if 'per_class' not in metric_name}


def to_int_tuple(my_tuple: tuple) -> tuple:

    return tuple(np.round(my_tuple).astype('int'))


def display_image_with_matplot(image:np.ndarray, title:str) -> None:

    if len(image.shape) < 4:
        exhibited_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        exhibited_image = np.squeeze(image, axis=0)
        exhibited_image = cv2.cvtColor(exhibited_image, cv2.COLOR_BGR2RGB)

    plt.axis("off")
    plt.title("{}".format(title))
    plt.imshow(exhibited_image.astype(np.uint8))
    plt.show()


def save_one_image(img_path: str, img_arr: np.ndarray) -> None:

    cv2.imwrite(img_path, np.squeeze(img_arr, axis=0),
                [cv2.IMWRITE_JPEG_QUALITY, 99])


def load_one_image(img_path: str, resize_dim: tuple = None) -> np.ndarray:

    img = cv2.imread(img_path)

    if resize_dim and len(resize_dim) == 2:
        x_dim, y_dim = resize_dim
        img = cv2.resize(img, dsize=(x_dim, y_dim),
                         interpolation=cv2.INTER_CUBIC)
    return img


class AdvImgCreator:

    def __init__(self, img_root: str, output_root: str) -> None:

        self.img_root = img_root
        ann_files = [p for p in os.listdir(self.img_root) if re.search(".json", p) is not None]
        Path(output_root).mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        if len(ann_files) > 0:
            for ann_file in ann_files:
                shutil.copyfile(os.path.join(self.img_root,ann_file), os.path.join(self.output_root, ann_file))

    def create_img(self, attack_method:Callable, raw_image_name:str, out_image_name:str, verbose=False) -> None:

        img_path = os.path.join(self.img_root, raw_image_name)
        img_arr = load_one_image(img_path)
        self.img_arr = np.expand_dims(img_arr, axis=0)
        output_path = os.path.join(self.output_root, out_image_name)
        try:
            self.atk_img_arr = attack_method.generate(x=self.img_arr, y=None)
        except Exception as e:
            print('error occur at %s with message %s' % (output_path, e))
        if self.atk_img_arr.shape[0] == 1 and len(self.atk_img_arr.shape) == 4:
            cv2.imwrite(output_path, np.squeeze(self.atk_img_arr,
                        axis=0), [cv2.IMWRITE_JPEG_QUALITY, 99])
            if verbose:
                print("The resulting maximal difference in pixel values is {}.".format(
                    np.amax(np.abs(self.img_arr - self.atk_img_arr))))
        else:
            raise ValueError('The resuting image dimension has to be 1*w*h*c')


class ExtractDtection:

    def __init__(self, prob_threshold: float = 0.5, all_class_list: list = COCO_INSTANCE_CATEGORY_NAMES) -> None:

        self.prob_threshold = prob_threshold
        self.all_class_list = all_class_list

    def extract_predictions(self, raw_predictions:Dict, print_predictions=False) -> Tuple[List, List]:
        # modified from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/application_object_detection.py
        # at line 103, def extract_predictions

        # todo: raise customized error
        assert (isinstance(raw_predictions, dict))
        raw_pred_labels = list(raw_predictions["labels"])
        raw_pred_boxes = list(raw_predictions['boxes'])
        # Get the predicted prediction score
        predictions_score = list(raw_predictions["scores"])

        # Get the predicted class
        predictions_class = [self.all_class_list[i] for i in raw_pred_labels]
        # Get the predicted bounding boxes
        predictions_boxes = [[(i[0], i[1]), (i[2], i[3])]
                             for i in raw_pred_boxes]
        if max(predictions_score) < 0.5:
            self.prob_threshold = np.quantile(np.array(predictions_score), 0.5)    
        # Get a list of index with score greater than threshold
        predictions_t = [predictions_score.index(
            x) for x in predictions_score if x > self.prob_threshold][-1]
        predictions_boxes = predictions_boxes[: predictions_t + 1]
        predictions_class = predictions_class[: predictions_t + 1]

        if print_predictions:
            print("predicted classes:", predictions_class)
            print("\npredicted score:", predictions_score[: predictions_t + 1])
        return predictions_class, predictions_boxes

    def output_formated_pred_list(self, raw_predictions:Dict) ->List[Tuple[List, List]]:
        return [self.extract_predictions(yi) for yi in raw_predictions]


class SavePlotWithPrediction:

    def __init__(self, pre_train_model:Callable, extractor:ExtractDtection, save_root:str) -> None:

        self.pre_train_model = pre_train_model
        self.extractor = extractor
        self.save_root = save_root
        Path(self.save_root).mkdir(parents=True, exist_ok=True)

    def display_model_prediction(self, image:np.ndarray, save_img_name:str, plot_box_setting: tuple) -> None:
        # modified from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/application_object_detection.py
        # within line 163 to 179

        if len(image.shape) == 4:
            img_arr = image.copy()
        elif len(image.shape) < 4:
            img_arr = np.expand_dims(image, axis=0)

        one_image_prediction = self.pre_train_model.predict(x=img_arr)[0]

        # Process predictions
        predictions_class, predictions_boxes = self.extractor.extract_predictions(
            one_image_prediction)
        # Plot predictions
        save_plot_path = os.path.join(self.save_root, save_img_name)
        plot_image_with_boxes(image.copy(), predictions_boxes,
                              predictions_class, plot_box_setting, save_plot_path)


def plot_image_with_boxes(img, boxes, pred_cls, plot_box_setting: tuple, save_plot_path: str) -> None:
    # modified from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/application_object_detection.py
    # at line 125, def plot_image_with_boxes

    if len(img.shape) < 4:
        plot_img = img.copy()
    else:
        plot_img = np.squeeze(img, axis=0)

    text_size, text_th, rect_th = plot_box_setting
    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(plot_img, to_int_tuple(boxes[i][0]), to_int_tuple(
            boxes[i][1]), color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(plot_img, pred_cls[i], to_int_tuple(
            boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    if save_plot_path:
        cv2.imwrite(save_plot_path, plot_img, [cv2.IMWRITE_JPEG_QUALITY, 99])
