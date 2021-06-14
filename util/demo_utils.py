import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from retina_face import config
from layers.functions.prior_box import PriorBox
from util.box_utils import decode, decode_landm
from util.align_faces import warp_and_crop_face, get_reference_facial_points
from eval_kit.detector_affect import EmotionDetector
from retina_face.retinaface import RetinaFace
from retina_face import cfg_mnet, cfg_re50
from util.nms.py_cpu_nms import py_cpu_nms

class DemoUtils():
    def __init__(self):     
        self.trained_model = './retina_face/mobilenet0.25_Final.pth'
        self.network = 'mobile0.25'
        cudnn.benchmark = True
        torch.set_grad_enabled(False)
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.8
        self.keep_top_k = 750
        self.resize = 1   
        self.cpu = False
        self.device = torch.device("cpu" if self.cpu else "cuda:0")
        self.AffectName = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']
        self.detector = EmotionDetector(fromPath = config.EMOTION_MODEL)
        self.cfg = None
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50
        self.net = RetinaFace(cfg = self.cfg, phase = 'test')
        self.net = self.load_model(self.net, self.trained_model, self.cpu)
        self.net.eval()
        self.net = self.net.to(self.device)

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        #print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        #print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def GetFacialPoints(self, img_raw):
        img = np.float32(img_raw)
        height, width, _ = img_raw.shape
        scale = torch.Tensor([width, height, width, height])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(height, width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        torch.cuda.empty_cache()
        return dets

    def GetRetinaROIs(self,image, confidence):
        faceConfidence = 0.0
        bestFaceConfidence = 0.0
        bestFace = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        threshedFaces = []

        try:
            gar=0
            # search every face in the image
            faces = self.GetFacialPoints(image)
            # return faces
            gar=0
            
            for i in range(len(faces)):
                # Each bounding box
                faceConfidence = faces[i][4]
                #print(faceConfidence)
                if faceConfidence>0.9:
                    threshedFaces.append(faces[i])
            return threshedFaces
            
        except:
            return None

    def Preprocess(self,image):
        confidence = 0.6
        height, width, _ = image.shape
        
        '''
        if height * width > 800000:
            image = cv2.resize(image, (int(width/4), int(height/4)))
            height, width, _ = image.shape
            print('New size:' + str(height) + ',' + str(width))
        '''
        
        try:
            # retinaROI
            retinas = self.GetRetinaROIs(image, confidence)
            # get the 5 landmarks of face
            facialPoints = []
            box_coords = []
            for face in retinas:
                box_coord = [face[0], face[1], face[2], face[3]]
                facialPts = [face[5], face[7], face[9], face[11], face[13], 
                        face[6], face[8], face[10], face[12], face[14]]
                facialPts = np.reshape(facialPts, (2, 5))
                box_coords.append(box_coord)
                facialPoints.append(facialPts)
            '''
            facialPoints = [retina[5], retina[7], retina[9], retina[11], retina[13], 
                        retina[6], retina[8], retina[10], retina[12], retina[14]]
            facialPoints = np.reshape(facialPoints, (2, 5))
            #print(facialPoints)
            '''
        except:
            return image

        # Parameters for alignment
        default_square = True
        output_size = (224, 224)
        inner_padding_factor = 0.05
        outer_padding = (0, 0)

        # get the reference 5 landmarks position in the crop settings
        referencePoint = get_reference_facial_points(output_size, inner_padding_factor, outer_padding, default_square)
        #print(referencePoint)

        #alignment
        alignedImgs = []
        for idx,face in enumerate(facialPoints):
            alignedImg = warp_and_crop_face(image, face, reference_pts = referencePoint, crop_size = output_size)
            '''
            box_coord = box_coords[idx]
            x1 = int(box_coord[0])
            x2 = int(box_coord[2])
            y1 = int(box_coord[1])
            y2 = int(box_coord[3])
            alignedImg = image[y1:y2,x1:x2,:]
            '''
            alignedImgs.append(alignedImg)
            # cv2.imwrite(str(idx)+".jpg", alignedImg)
        return alignedImgs,facialPoints,box_coords

    def recognizeEmotion(self, roiFaces, irOffset=False):
        probs = self.detector.predict(roiFaces)
        if irOffset:
            probs[:,0] += 0.15 #Neutral
            probs[:,1] += 0.1 #Happy
            probs[:,2] += 0.15 #Sad
            probs[:,3] -= 0.3 #Surprise
            probs[:,4] -= 0.35 #Fear
            probs[:,5] += 0.2 #Disgust
            probs[:,6] += 0.05 #Angry
        emo_idx = np.argmax(probs,axis=1)
        emotions = [self.AffectName[int(key)] for key in emo_idx]
        return emotions,probs

    def _write_row(self, df, frame_cnt, face_coordinates, emotions, probs):
        # we can calculate the people count by counting the length of emotions
        # columns=["people_cnt", "people_idx", "frame", "emotion"]
        for idx, emotion in enumerate(emotions):
            x0, y0, x1, y1 = face_coordinates[idx]
            df = df.append({'people_cnt' : len(emotions),
                            'people_idx' : idx , 
                            # 'name' : names[idx] , 
                            'frame' : frame_cnt, 
                            'emotion' : emotion,
                            "x0": "{:.2f}".format(x0),
                            "y0": "{:.2f}".format(y0),
                            "x1": "{:.2f}".format(x1),
                            "y1": "{:.2f}".format(y1),
                            "Angry": "{:.4f}".format(probs[idx][6]),
                            "Disgust": "{:.4f}".format(probs[idx][5]),
                            "Fear": "{:.4f}".format(probs[idx][4]),
                            "Happy": "{:.4f}".format(probs[idx][1]),
                            "Neutral": "{:.4f}".format(probs[idx][0]),
                            "Sad": "{:.4f}".format(probs[idx][2]),
                            "Surprise": "{:.4f}".format(probs[idx][3]),
                            "Contempt": "{:.4f}".format(probs[idx][7]),
                            } , ignore_index=True)
        return df

    def draw_faces_and_emotion(self, im, bboxes, emotions, probs):
        output = im.copy()
        font = cv2.FONT_HERSHEY_DUPLEX
        alpha = 0.5
        color_code = {
                'Neutral': (243, 223, 191), # skin
                'Happy': (66, 226, 184), # green
                'Sad': (75, 150, 203), # orange
                'Surprise': (75, 150, 203), # orange
                'Fear': (235, 138, 144), # red
                'Disgust': (243, 223, 191), # skin
                'Angry': (235, 138, 144), # red
                'Contempt': (243, 223, 191), # skin
                }
        print(f"len(bboxes) = {len(bboxes)} | len(emotions) = {len(emotions)}")
        print(emotions)        
        for bbox, emotion in zip(bboxes, emotions):
            x0, y0, x1, y1 = [int(_) for _ in bbox]
            # im = cv2.putText(im, emotion, (x0-20, y0-5), font, 1, color_code[emotion], 1)
            # emotion bg
            im = cv2.rectangle(im, (x0-20, y0-30), (x0+len(emotion)*15, y0), (118,115,112), cv2.FILLED)
            # name bg
            # im = cv2.rectangle(im, (x0, y1+15), (x0+len(name)*13, y1+45), (42, 219, 151), cv2.FILLED)
            # im = cv2.putText(im, name, (x0, y0+140), font, 1, (255,255,255), 1)    
        cv2.addWeighted(im, alpha, output, 1 - alpha, 0, output)
        for bbox, emotion, prob in zip(bboxes, emotions, probs):
            # face roi
            x0, y0, x1, y1 = [int(_) for _ in bbox]
            output = cv2.rectangle(output, (x0, y0), (x1, y1), color_code[emotion], 2)
            x0, y0, x1, y1 = [int(_) for _ in bbox]
            emo_idx = np.argmax(prob)
            output = cv2.putText(output, emotion+" : "+ str(prob[emo_idx]), (x0-20, y0-5), font, 1, color_code[emotion], 1, cv2.LINE_AA)
            # output = cv2.putText(output, name, (x0, y1+40), font, 0.7, (255,255,255), 1, cv2.LINE_AA)    

        return output