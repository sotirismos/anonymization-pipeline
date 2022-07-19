# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import glob as glob
import numpy as np
from utilities import blur_img_bboxes
from mmdet.apis import init_detector, inference_detector

def two_stage_lp(img_paths, settings_vehicle, vehicle_thresh, settings_lp, lp_thresh):
    
    # build the model from a config file and a checkpoint file
    model_vehicle = init_detector(settings_vehicle['config_path'], settings_vehicle['model_path'], device='cuda:0')
    model_lp = init_detector(settings_lp['config_path'], settings_lp['model_path'], device='cuda:0')

    detections_image = []
    for in_img_path in img_paths:
        detections = []
        img = cv2.imread(in_img_path)
        dets = inference_detector(model_vehicle, img)
        cropped_vehicles = []
        for bbox in dets[2]:
            if bbox[4] >= vehicle_thresh:
                cropped_vehicles += [(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])]
        for bbox in dets[3]:
            if bbox[4] >= vehicle_thresh:
                cropped_vehicles += [(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])] 
        for bbox in dets[4]:
            if bbox[4] >= vehicle_thresh:
                cropped_vehicles += [(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])] 
        for bbox in dets[6]:
            if bbox[4] >= vehicle_thresh:
                cropped_vehicles += [(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])], [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])]
               
                
        # license plate detection
        for i, vehicle in enumerate(cropped_vehicles):
            dets = inference_detector(model_lp, vehicle[0])
            for bbox in dets[0]:
                if bbox[4] >= lp_thresh:
                    tmp = [{'name': 'LicensePlate',
                           'percentage_probability': bbox[4],
                           'box_points': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]}]
                    tmp[0]['box_points'] = np.asarray(tmp[0]['box_points'] + np.tile(vehicle[1][:2], 2)).tolist()
                    detections += tmp
        detections_image.append(detections)
        
    return detections_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer two models in parallel')
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
                        help="a list of paths to json configuration files (1.vehicle detection, 2.lp detection)"
                        )
    parser.add_argument('--model',
                        nargs=2,
                        dest='model_list',
                        required=True,
                        help="a list of paths to the checkpoint files (1.vehicle detection, 2.lp detection)"
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
                        default=0.1,
                        help="lp detection threshold"
                        )
    
    args = parser.parse_args()

    img_dir = args.img_dir
    out_dir = args.out_dir
    model_list = args.model_list
    cfg_list = args.cfg_list
    vehicle_thresh = args.vehicle_thresh
    lp_thresh = args.lp_thresh
    
    print(model_list)
    
    for i in range(len(model_list)):
        assert os.path.isfile(model_list[i]), "Not a valid model file %s"\
            % (model_list[i])
        assert os.path.isfile(cfg_list[i]), "Not a valid model file %s"\
            % (cfg_list[i])

    for i in range(len(model_list)):
        assert os.path.isfile(model_list[i]), "Not a valid model file %s"\
            % (model_list[i])
        assert os.path.isfile(cfg_list[i]), "Not a valid model file %s"\
            % (cfg_list[i])

    settings_vehicle = {"model_path": os.path.abspath(args.model_list[0]),
                     "config_path": os.path.abspath(args.cfg_list[0]),
                     "threshold": vehicle_thresh}
    settings_lp = {"model_path": os.path.abspath(args.model_list[1]),
                    "config_path": os.path.abspath(args.cfg_list[1]),
                    "threshold": lp_thresh}

    input_paths = glob.glob(os.path.join(img_dir, "*.png"))    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    detections_lp = two_stage_lp(input_paths, settings_vehicle, vehicle_thresh, settings_lp, lp_thresh)
    
    for i in range(len(input_paths)):
        blur_img_bboxes(input_paths[i], detections_lp[i], out_dir)