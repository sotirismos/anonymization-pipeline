"""
@author: sotiris
"""

import os
import pickle
import cv2
import argparse
import glob as glob
import numpy as np
from utilities import denormalize, match, draw_img_bboxes

def two_stage_lp(img_paths, settings_car, settings_lp):
    from imageai.Detection.Custom import CustomObjectDetection

    detector_car = CustomObjectDetection()
    detector_car.setModelTypeAsYOLOv3()
    detector_car.setModelPath(settings_car["model_path"])
    detector_car.setJsonPath(settings_car["config_path"])
    detector_car.loadModel()

    detector_lp = CustomObjectDetection()
    detector_lp.setModelTypeAsYOLOv3()
    detector_lp.setModelPath(settings_lp["model_path"])
    detector_lp.setJsonPath(settings_lp["config_path"])
    detector_lp.loadModel()


    crop_image = lambda img, bb: img[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]

    detections_image = []
    for in_img_path in img_paths:
        detections = []
        img = cv2.imread(in_img_path)
        tmp = detector_car.detectObjectsFromImage(input_image=in_img_path,
                                                      output_type="array",
                                                      minimum_percentage_probability=\
                                                          settings_car["threshold"])[1]

        rois = [(crop_image(img, det["box_points"]), det["box_points"]) \
                 for det in tmp if det["name"] in ("car","truck","motorbike","bus")]

        for i, roi in enumerate(rois):
            tmp = detector_lp.detectObjectsFromImage(input_image=roi[0],
                                                     input_type="array",
                                                      output_type="array",
                                                      minimum_percentage_probability=\
                                                    settings_lp["threshold"])[1]
            for det_lp in tmp:
                det_lp["box_points"] = (np.asarray(det_lp["box_points"]) +\
                    np.tile(roi[1][:2], 2)).tolist()
            
            detections += tmp
        detections_image.append(detections)

    return detections_image

def make_inputs(anno_dir, img_dir):
    anno_file_paths = glob.glob(os.path.join(anno_dir, "*.txt"))
    in_img_list = []
    bboxes = []
    for anno_file_path in anno_file_paths:
        in_img_list +=\
            [os.path.join(img_dir,os.path.split(os.path.splitext(anno_file_path)[0])[-1]+".jpg")]
        anno_file = open(anno_file_path)
        objects = anno_file.readlines()                                       # read all lines into a list
        img = cv2.imread(in_img_list[-1])                                     # read image
        if img is None:
            print("corrupted image file")
            in_img_list.pop()
            continue
        else:
            bboxes.append([denormalize(obj, img.shape[:-1])
                           for obj in objects if obj.split()[0] in ('15')])
    return in_img_list, bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer two models in parallel')
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
                        nargs=2,
                        dest='cfg_list',
                        required=True,
                        help="a list of paths to json configuration files"
                        )
    parser.add_argument('--model',
                        nargs=2,
                        dest='model_list',
                        required=True,
                        help="a list of paths to the model files"
                        )
    parser.add_argument('--out_dir',
                        dest='out_dir',
                        type=str,
                        required=False,
                        default=os.getcwd(),
                        help="path to the output directory"
                        )
    parser.add_argument('--vehicle_thresh',
                        dest='vehicle_thresh',
                        type=float,
                        required=False,
                        default=0.3,
                        help="vehicle detection threshold"
                        )
    parser.add_argument('--lp_thresh',
                        dest='lp_thresh',
                        type=float,
                        required=False,
                        default=0.5,
                        help="lp detection threshold"
                        )
    
    args = parser.parse_args()

    anno_dir = os.path.abspath(args.anno_dir)
    img_dir = os.path.abspath(args.img_dir)
    out_dir = os.path.abspath(args.out_dir)
    vehicle_thresh = 100*args.vehicle_thresh
    lp_thresh = 100*args.lp_thresh

    input_paths, bboxes = make_inputs(anno_dir, img_dir)

    for i in range(len(args.model_list)):
        assert os.path.isfile(args.model_list[i]), "Not a valid model file %s"\
            % (args.model_list[i])
        assert os.path.isfile(args.cfg_list[i]), "Not a valid model file %s"\
            % (args.cfg_list[i])

    for i in range(len(args.model_list)):
        assert os.path.isfile(args.model_list[i]), "Not a valid model file %s"\
            % (args.model_list[i])
        assert os.path.isfile(args.cfg_list[i]), "Not a valid model file %s"\
            % (args.cfg_list[i])

    settings_orig = {"model_path": os.path.abspath(args.model_list[0]),
                     "config_path": os.path.abspath(args.cfg_list[0]),
                     "threshold": vehicle_thresh}
    settings_cus = {"model_path": os.path.abspath(args.model_list[1]),
                    "config_path": os.path.abspath(args.cfg_list[1]),
                    "threshold": lp_thresh}

    out_dir_path = os.path.abspath(os.path.join(os.getcwd(), out_dir))
    if not os.path.isdir(out_dir_path):
        os.mkdir(out_dir_path)
    detections_lp = two_stage_lp(input_paths, settings_orig, settings_cus)
    
    mdict_list = []
    for i in range(len(input_paths)):
        draw_img_bboxes(input_paths[i], detections_lp[i], out_dir_path)
        mdict = dict( )
        mdict["file_path"] = input_paths[i]
        mdict["matches"] = match(detections_lp[i], bboxes[i])
        mdict["det"] = detections_lp[i]
        mdict["gt"] = [ dict({"bbox": bbox, "class": r"LicensePlate"}) for bbox in bboxes[i]]
        mdict_list.append(mdict)
    pickle.dump(mdict_list, open(r"two_stage_mdict_list.pkl", "wb"))

