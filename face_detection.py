"""
@author: sotiris
"""
import os
import cv2
import natsort
import argparse
from glob import glob
import pickle

from deepface.detectors import FaceDetector

def blur_face(img, bbox):
    x, y, w, h = bbox
    roi = img[y:h, x:w]
    
    # applying a gaussian blur over this new rectangle area
    roi = cv2.GaussianBlur(roi, (23, 23), 30)
    
    # impose this blurred image on original image to get final image
    img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        
    return img

def face_detection(frames, save_dir, backend, threshold):
    face_detector = FaceDetector.build_model(backend)
    print("Detector backend is ", backend)
    
    save_path_backend = os.path.join(save_dir, backend)                         
    os.makedirs(save_path_backend, exist_ok=True)                             # Create Directory depending on backend
    
    detected_faces = {}            
    frame_idx = 1
    for frame in natsort.natsorted(frames):                                   # Loop over all sorted frames of a particular quality of a video        
        img = cv2.imread(frame)
        faces = FaceDetector.detect_faces(face_detector, backend, img, align = False)
        detected_faces[frame] = {}
        detected_faces[frame]['frame_idx'] = []
        detected_faces[frame]['face_id'] = []
        detected_faces[frame]['image_path'] = []
        detected_faces[frame]['bbox'] = []
        detected_faces[frame]['confidence'] = []
        
        if len(faces) == 0:
            frame_idx += 1
            continue
                                                         
        face_id = 0    
        for (face, img_region, confidence) in faces:
            if (confidence >= threshold):
                detected_faces[frame]['frame_idx'] = frame_idx
                detected_faces[frame]['face_id'].append(face_id)
                image_path = os.path.join(backend +'/' + str(frame_idx) + '/' + str(face_id) + '.png')
                detected_faces[frame]['image_path'].append(image_path)
                detected_faces[frame]['bbox'].append(img_region)
                detected_faces[frame]['confidence'].append(confidence)
                img = blur_face(img, detected_faces[frame]['bbox'][face_id])
                face_id += 1
        cv2.imwrite(f"{save_path_backend}/{frame_idx}_{face_id - 1}.png", img)
        frame_idx += 1
    
    # store detections as pickle file
    with open(os.path.join(save_path_backend,'face_detections.pickle'), 'wb') as handle:
        pickle.dump(detected_faces, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and blur faces in subsequent frames')
    parser.add_argument('--frames_dir',
                        dest='frames',
                        type=str,
                        required=True,
                        help='Directory containing the desired frame(s)'
                        )
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        type=str,
                        required=True,
                        help='Root directory to store the frame(s) after detection and blurring of the faces and the pickle file containing the information of the detections'
                        )
    args = parser.parse_args()
    
    frames = glob(os.path.join(args.frames, '*')) # Get the path(s) of the desired frame(s)
    backend = 'retinaface' # Backend method for face detection
    threshold = 0.9        # Threshold value for face detection (for retinaface the threshold is 0.9 by default)
                
    face_detection(frames, args.save_dir, backend, threshold)
