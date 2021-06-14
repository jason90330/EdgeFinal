import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import time
from util.demo_utils import DemoUtils
from retina_face import config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

if __name__ == '__main__':
    retina_utils = DemoUtils()    
    # num_img = len(imagePaths)
    # print("Total :" + str(num_img) + '\n')
    
    cap = cv2.VideoCapture(config.VPATH)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    # Drop the input path with ".mp4"
    out_filename = config.VPATH[:-4]
    # initialize mp4 writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    output_movie = cv2.VideoWriter(out_filename + "_out.mp4", fourcc, fps, (width, height))
    # df = pd.DataFrame(columns=["people_cnt", "people_idx", "name", "frame", "emotion", "x0", "y0", "x1", "y1", "Angry", 
    #                             "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"])
    df = pd.DataFrame(columns=["people_cnt", "people_idx", "frame", "emotion", "x0", "y0", "x1", "y1", 
                                "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt"])
    print(f"v_path ={config.VPATH}")
    print("cap.isOpened() =", cap.isOpened())
    skip_count = 0
    
    # For args.skip_frames, To achieve real-time demo, we reducing the processing frame
    if fps > config.RUNING_FPS:
        skip_count = int(fps / config.RUNING_FPS) -1
        print(f"Processing {int(fps / config.RUNING_FPS)} frames each seconds")
    boxes = []
    emotions = []
    cnt = 1
    # detector = EmotionDetector(fromPath = config.EMOTION_MODEL)
    # AffectName = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

    while(cap.isOpened()):
        t = time.time()
        ret, frame = cap.read()

        if ret:
            roiFaces,facialPoints,box_coords = retina_utils.Preprocess(frame)

            emotions, probs = retina_utils.recognizeEmotion(roiFaces)            

            df = retina_utils._write_row(df, cnt, box_coords, emotions, probs)

            # Draw bounding box for visiualization
            frame = retina_utils.draw_faces_and_emotion(frame, box_coords, emotions)
            output_movie.write(frame)
            if config.GUI_MODE:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1)  & 0xFF == ord('q'):
                    break
            print(f"Overall runtime each frame: {time.time()- t:.3f}")
            print('FPS {:1f}'.format(1/(time.time() -t)))
            cnt += 1
        else:
            break
    df.to_csv(out_filename + ".csv", index=False)
    output_movie.release()
    cap.release()
    print(f"Finish {out_filename} and write to csv")