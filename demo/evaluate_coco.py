import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def evaluate_coco(gt_json_path, preds_dict):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(preds_dict["annotations"])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")  # or "bbox"
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# Example usage
if __name__ == "__main__":
    # Load your predictions (as dict)
    preds_dict = json.load(
        open(
            "/home/jclyons/endovis/challenge_2025/MaskDINO/demo/outputs/subchallengeC.json"
        )
    )

    # Run evaluation
    evaluate_coco(
        "/media/SSD3/leoshared/Dataset/cvs_challenge25_validation/data/annotations/annotations_C.json",
        preds_dict,
    )
