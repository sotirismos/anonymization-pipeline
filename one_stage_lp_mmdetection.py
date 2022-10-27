# -*- coding: utf-8 -*-
import os
import pickle
import cv2
import argparse
import glob as glob
from utilities import denormalize, match
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


def make_inputs(anno_dir, img_dir):
    anno_file_paths = glob.glob(os.path.join(anno_dir, "*.txt"))
    in_img_list = []
    bboxes = []
    for anno_file_path in anno_file_paths:
        in_img_list += \
            [os.path.join(img_dir, os.path.split(os.path.splitext(anno_file_path)[0])[-1] + ".jpg")]
        anno_file = open(anno_file_path)
        objects = anno_file.readlines()  # read all lines into a list
        img = cv2.imread(in_img_list[-1])  # read image
        if img is None:
            print("corrupted image file")
            in_img_list.pop()
            continue
        else:
            bboxes.append([denormalize(obj, img.shape[:-1])
                           for obj in objects if obj.split()[0] in ('15')])
    return in_img_list, bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='one stage lp detection')
    parser.add_argument('--anno_dir',
                        dest='anno_dir',
                        type=str,
                        required=True,
                        help="path to annotation directory"
                        )
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

    anno_dir = args.anno_dir
    img_dir = args.img_dir
    out_dir = args.out_dir
    model_path = args.model_path
    cfg_path = args.cfg_path
    lp_thresh = args.lp_thresh

    input_paths, bboxes = make_inputs(anno_dir, img_dir)

    assert os.path.isfile(cfg_path[0]), "Not a valid cfg file %s" % cfg_path
    assert os.path.isfile(model_path[0]), "Not a valid model file %s" % model_path

    settings_lp = {"model_path": os.path.abspath(args.model_path[0]),
                   "config_path": os.path.abspath(args.cfg_path[0]),
                   "threshold": lp_thresh}

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    detections_lp = one_stage_lp(input_paths, settings_lp, lp_thresh)

    mdict_list = []
    for i in range(len(input_paths)):
        mdict = dict()
        mdict["file_path"] = input_paths[i]
        mdict["matches"] = match(detections_lp[i], bboxes[i])
        mdict["det"] = detections_lp[i]
        mdict["gt"] = [dict({"bbox": bbox, "class": r"LicensePlate"}) for bbox in bboxes[i]]
        mdict_list.append(mdict)
    pkl_file_outpath = os.path.join(out_dir, 'mdict_list_mmdetection.pkl')
    with open(pkl_file_outpath, 'wb') as handle:
        pickle.dump(mdict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
