import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import time
from util.demo_utils import DemoUtils
from retina_face import config

if __name__ == '__main__':
    retina_utils = DemoUtils()    
    video_capture = cv2.VideoCapture(0)
    # detector = EmotionDetector()
    process_this_frame = True
    box_coords = []
    emotions = []
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            t = time.time()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            roiFaces,facialPoints,box_coords = retina_utils.Preprocess(frame)
            if len(roiFaces)>0:
                emotions, probs = retina_utils.recognizeEmotion(roiFaces)
                # probs = retina_utils.detector.predict(roiFaces)
                # emo_idx = np.argmax(probs,axis=1)
                # emotions = [retina_utils.AffectName[int(key)] for key in emo_idx]

        frame = retina_utils.draw_faces_and_emotion(frame, box_coords, emotions, probs)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()