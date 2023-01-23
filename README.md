# Adversarial_ML
Demos of adversarial maching learning in computer vision.

# Build the enviroment with poetry
curl -sSL https://install.python-poetry.org | python3 -

mkdir [dirname]

cd [dirname]

poetry init	# use this dir to create the project

poetry config virtualenvs.in-project true #create the env of the project in this dir; the .env folder will be in this dir

poetry shell # Now that the env is in this dir, one can activate the env by enter poetry shell in the terminal under this dir

poetry add numpy

poetry add torch==1.12.1

poetry add torchvision==0.13.1

poetry add torchmetrics==0.9.3

poetry add pycocotools

poetry add opencv-python

poetry add adversarial-robustness-toolbox

poetry add matplotlib

# The quick start with the Google colab
copy the following commands in any cell and the execute it

%%shell

pip install virtualenv --quiet

virtualenv --version

mkdir /content/myenv

cd /content/myenv

virtualenv advml

source /content/myenv/advml/bin/activate

pip install torch==1.12.1 --quiet

pip install torchvision==0.13.1 --quiet

pip install torchmetrics==0.9.3 --quiet

pip install pycocotools --quiet

pip install opencv-python --quiet

pip install adversarial-robustness-toolbox --quiet

pip install matplotlib --quiet

pip install ipykernel --quiet

# Example 1: 
## Assuming using ./example_imgs/raw_image to create adversarial images, one would like to apply the PGD attack where source model and the victim model is same.
### 1. cd to the dir where the env is established
### 2. open the terminal and enter poetry shell
### 3. enter the following
#### python run_gradient_adv.py --source_model ssd300_vgg16 --img_type jpg --raw_image_root ./example_imgs/raw_image --adv_image_root ./example_imgs/adv_image --display_raw_pred_root ./example_imgs/raw_pred --display_adv_pred_root ./example_imgs/adv_pred --atk_eps 270 --atk_eps_step 2 --atk_max_iter 5

# Example 2: 
## Assuming using ./example_imgs/raw_image to create adversarial images, one would like to apply the PGD attack where source model and the victim model is different.
### 1. cd to the dir where the env is established
### 2. open the terminal and enter poetry shell
### 3. enter the following
#### python run_gradient_adv.py --source_model ssd300_vgg16 --victim_model ssdlite320_mobilenet_v3_large --img_type jpg --raw_image_root ./example_imgs/raw_image --adv_image_root ./example_imgs/adv_image --display_raw_pred_root ./example_imgs/raw_pred --display_adv_pred_root ./example_imgs/adv_pred --atk_eps 270 --atk_eps_step 2 --atk_max_iter 5

# Example 3:
## Test if the victim model performance is affected by the PGD attack, assuming that source model and the victim model is different.
### 1. cd to the dir where the env is established
### 2. open the terminal and enter poetry shell
### 3. enter the following to create adversarial images for the sample COCO data set
#### python run_gradient_adv.py --source_model ssd300_vgg16 --victim_model ssdlite320_mobilenet_v3_large --img_type jpg --raw_image_root ./sample_COCO_val_2017_data --adv_image_root ./sample_adv_coco_imgs --display_raw_pred_root ./coco_raw_pred --display_adv_pred_root ./coco_adv_pred --atk_eps 270 --atk_eps_step 2 --atk_max_iter 5
### 4. enter the following to caculate the model performance on the original data set "/sample_COCO_val_2017_data"
#### python run_detection_performance.py --model_name ssdlite320_mobilenet_v3_large --coco_img_root ./sample_COCO_val_2017_data
### 5. enter the following to caculate the model performance on the original data set "/sample_adv_coco_imgs"
#### python run_detection_performance.py --model_name ssdlite320_mobilenet_v3_large --coco_img_root ./sample_adv_coco_imgs

# Note:
1. The package currently cannot use "ssdlite320_mobilenet_v3_large" as the source model.
2. YOLO models hasn't been included.
3. Use models listed in adv_lib/__init__.py at line 10 model_weights.keys()
