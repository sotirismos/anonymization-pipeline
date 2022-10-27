# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import glob as glob
import numpy as np
from utilities import blur_img_bboxes
from mmdet.apis import init_detector, inference_detector


def one_stage_lp(img_paths, settings_lp, lp_thresh):
    # build the model from a config file and a checkpoint file
    model_lp = init_detector(settings_lp['config_path'], settings_lp['model_path'], device='cuda:0')

    detections_image = []
    for in_img_path in img_paths:
        detections = []
        img = cv2.imread(in_img_path)

        # license plate detection
        dets = inference_detector(model_lp, img)
        for bbox in dets[0]:
            if bbox[4] >= lp_thresh:
                tmp = [{'name': 'LicensePlate',
                        'percentage_probability': bbox[4],
                        'box_points': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]}]
                detections += tmp
        detections_image.append(detections)

    return detections_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='one stage lp detection')
    parser.add_argument('--img_dir',
                        dest='img_dir',
                        type=str,
                        required=True,
                        help="path to images directory"
                        )
    parser.add_argument('--config',
                        nargs=1,
                        dest='cfg_path',
                        required=True,
                        help="a path to the configuration file"
                        )
    parser.add_argument('--model',
                        nargs=1,
                        dest='model_path',
                        required=True,
                        help="a path to the checkpoint file"
                        )
    parser.add_argument('--out_dir',
                        dest='out_dir',
                        type=str,
                        required=False,
                        default=os.getcwd(),
                        help="path to the output directory"
                        )
    parser.add_argument('--lp_thresh',
                        dest='lp_thresh',
                        type=float,
                        required=False,
                        default=0.1,
                        help="lp detection threshold"
                        )

    args = parser.parse_args()

    img_dir = args.img_dir
    out_dir = args.out_dir
    model_path = args.model_path
    cfg_path = args.cfg_path
    lp_thresh = args.lp_thresh

    print(model_path)
    assert os.path.isfile(cfg_path[0]), "Not a valid cfg file %s" % cfg_path
    assert os.path.isfile(model_path[0]), "Not a valid model file %s" % model_path

    settings_lp = {"model_path": os.path.abspath(args.model_path[0]),
                   "config_path": os.path.abspath(args.cfg_path[0]),
                   "threshold": lp_thresh}

    input_paths = glob.glob(os.path.join(img_dir, "*.png"))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    detections_lp = one_stage_lp(input_paths, settings_lp, lp_thresh)

    for i in range(len(input_paths)):
        blur_img_bboxes(input_paths[i], detections_lp[i], out_dir)
