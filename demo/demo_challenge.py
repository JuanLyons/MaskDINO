# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import os
import pycocotools.mask as mask_util

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm
import json
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from detectron2.engine.defaults import DefaultPredictor


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/cvs/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_fold1.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


class Demo(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
        """
        predictions = self.predictor(image)

        return predictions


class Predictions(object):
    def __init__(self, categories):
        self.predictions_json = {"images": [], "annotations": [], "categories": []}
        self.update_categories(categories)
        self.annotation_id = 1

    def update_categories(self, categories):
        for category in categories:
            category_id = category["id"]
            category_name = category["name"]
            category_dict = {"id": category_id, "name": category_name}
            self.predictions_json["categories"].append(category_dict)

    def update_images(self, images):
        for image in images:
            file_name = image["file_name"]
            height = image["height"]
            width = image["width"]
            image_id = image["id"]
            image_dict = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": image_id,
            }
            self.predictions_json["images"].append(image_dict)

    def get_images(self):
        return self.predictions_json["images"]

    def update_annotations(self, image, predictions):
        instances = predictions["instances"].to("cpu")
        for idx in range(len(instances)):
            prediction = instances[idx]

            # Get bbox
            x1, y1, x2, y2 = prediction.pred_boxes.tensor.numpy()[0]
            x, y, w, h = round(x1), round(y1), round(x2 - x1), round(y2 - y1)

            # Get segmentation mask
            mask = prediction.pred_masks[0]
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8")
            )[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            annotation_dict = {
                "id": self.annotation_id,
                "image_id": image["id"],
                "height": image["height"],
                "width": image["width"],
                "bbox": [x, y, w, h],
                "category_id": int(prediction.pred_classes.item()) + 1,
                "score": float(prediction.scores),
                "segmentation": rle,
            }

            self.predictions_json["annotations"].append(annotation_dict)

            self.annotation_id += 1

    def get_predictions_json(self):
        return self.predictions_json


if __name__ == "__main__":
    args = get_parser().parse_args()

    data_path = os.environ.get("DATA_PATH")
    res_path = os.environ.get("RES_PATH")
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = Demo(cfg)

    json_file = load_json(os.path.join(data_path, "metadata", "test_frames_C.json"))

    predictions = Predictions(json_file["categories"])
    predictions.update_images(json_file["images"])

    predictions_path = os.path.join(res_path, "subchallengeC.json")

    for frame in tqdm(predictions.get_images()):
        path = os.path.join(data_path, "frames", frame["file_name"])
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        model_predictions = demo.run_on_image(img)

        predictions.update_annotations(frame, model_predictions)

    write_json(predictions_path, predictions.get_predictions_json())
