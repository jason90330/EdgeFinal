import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.backends.cudnn as cudnn
import time
import ctypes
import _ctypes
import pygame
import sys
from util.demo_utils import DemoUtils
from retina_face import config
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class InfraRedRuntime(object):
    def __init__(self):
        self.retina_utils = DemoUtils()  
        self._done = False
        self._clock = pygame.time.Clock()
        # self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        # Parameters
        # trained_model = './weights/Resnet50_Final.pth'
        # network = 'resnet50'
        trained_model = './weights/mobilenet0.25_Final.pth'
        network = 'mobile0.25'
        cpu = False
        cudnn.benchmark = True
        torch.set_grad_enabled(False)
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.resize = 1
        self.bright_scale = 1
        
    def norm_by_depth(self, rgb, depth):
        bright_scale = 30
        Depth_threshold = 8000
        depth_frame = np.float32(depth)
        depth_frame[depth_frame > Depth_threshold] = Depth_threshold  
        norm_scale = depth_frame / Depth_threshold * bright_scale
        rgb = np.multiply(rgb, norm_scale)
        return rgb

    def show_infrared_frame(self, frame, depth_frame):
        Infrared_threshold = 65535
        frame = np.float32(frame)
        image_infrared_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                            self._kinect.depth_frame_desc.Width])

        image_infrared_all = self.norm_by_depth(image_infrared_all, depth_frame)                                         
        # 转换为（n，m，1） 形式
        image_infrared_all = image_infrared_all * self.bright_scale
        image_infrared_all[image_infrared_all > Infrared_threshold] = Infrared_threshold        
        image_infrared_all = image_infrared_all / Infrared_threshold * 255
        image_infrared_all = np.uint8(image_infrared_all)
        result = infrared = image_infrared_all[:,::-1]
        return image_infrared_all

    def get_the_last_depth(self):
        """
        Time :2019/5/1
        FunC:获取最新的图像数据
        Input:无
        Return:无
        """
        if self._kinect.has_new_depth_frame():
            # 获得深度图数据
            frame = self._kinect.get_last_depth_frame()
            # 转换为图像排列
            image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height,
                                             self._kinect.depth_frame_desc.Width])
            self.depth_ori = image_depth_all

            return self.depth_ori

    def run(self):
        # FERPlusName = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']
        process_this_frame = True
        video_capture = cv2.VideoCapture(0)
        while not self._done:
            depth_frame = self.get_the_last_depth()
            '''
            if process_this_frame and depth_frame is not None:
                frame = self.show_depth_frame(depth_frame)
                cv2.imshow("depth_frame", frame)
                if cv2.waitKey(1)  & 0xFF == ord('q'):
                    break
            '''
            if self._kinect.has_new_infrared_frame() and process_this_frame:
                frame = self._kinect.get_last_infrared_frame()
                # ret, frame = video_capture.read()
                # ir_frame = frame
                try:
                    ir_frame = self.show_infrared_frame(frame, depth_frame)
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2RGB)
                    # cv2.imshow("ir_frame", ir_frame)
                    roiFaces,facialPoints,box_coords = self.retina_utils.Preprocess(ir_frame)
                    # result = self.draw_box(ir_frame, box_coords)
                    if len(roiFaces)>0:
                        emotions, probs = self.retina_utils.recognizeEmotion(roiFaces, irOffset=True)    
                        # result = ir_frame
                        result = self.retina_utils.draw_faces_and_emotion(ir_frame, box_coords, emotions, probs)
                except:
                    result = ir_frame
                    print("Non found")
                cv2.imshow('result', result)
                if cv2.waitKey(1)  & 0xFF == ord('q'):
                    break
            
            self._clock.tick(60)
        self._kinect.close()
        cv2.destroyAllWindows()


__main__ = "Kinect v2 InfraRed"
game =InfraRedRuntime();
game.run();