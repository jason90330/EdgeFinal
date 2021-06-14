import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.backends.cudnn as cudnn
import time
import pickle
import imutils
from imutils import paths

from retina_face import config
from retina_face import cfg_mnet, cfg_re50
from retina_face.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from util.nms.py_cpu_nms import py_cpu_nms
from util.box_utils import decode, decode_landm
from util.align_faces import warp_and_crop_face, get_reference_facial_points
from eval_kit.detector_affect import EmotionDetector

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def GetFacialPoints(img_raw):
    img = np.float32(img_raw)
    height, width, _ = img_raw.shape
    scale = torch.Tensor([width, height, width, height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    torch.cuda.empty_cache()
    return dets

def GetRetinaROIs(image, confidence):
    faceConfidence = 0.0
    bestFaceConfidence = 0.0
    bestFace = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    threshedFaces = []

    try:
        gar=0
        # search every face in the image
        faces = GetFacialPoints(image)
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

def Preprocess(image):
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
        retinas = GetRetinaROIs(image, confidence)
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

def recognizeEmotion(roiFaces):
    probs = detector.predict(roiFaces)
    emo_idx = np.argmax(probs,axis=1)
    emotions = [AffectName[int(key)] for key in emo_idx]
    return emotions,probs

def _write_row(df, frame_cnt, face_coordinates, emotions, probs):
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

def draw_faces_and_emotion(im, bboxes, emotions):
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
    for bbox, emotion in zip(bboxes, emotions):
        # face roi
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        output = cv2.rectangle(output, (x0, y0), (x1, y1), color_code[emotion], 2)
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        output = cv2.putText(output, emotion, (x0-20, y0-5), font, 1, color_code[emotion], 1, cv2.LINE_AA)
        # output = cv2.putText(output, name, (x0, y1+40), font, 0.7, (255,255,255), 1, cv2.LINE_AA)    

    return output

# Parameters
# trained_model = './weights/Resnet50_Final.pth'
# network = 'resnet50'
trained_model = './retina_face/mobilenet0.25_Final.pth'
network = 'mobile0.25'
cpu = False
cudnn.benchmark = True
torch.set_grad_enabled(False)
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.8
keep_top_k = 750
resize = 1

cfg = None
if network == "mobile0.25":
    cfg = cfg_mnet
elif network == "resnet50":
    cfg = cfg_re50
# net and model
net = RetinaFace(cfg = cfg, phase = 'test')
net = load_model(net, trained_model, cpu)

net.eval()
device = torch.device("cpu" if cpu else "cuda:0")
net = net.to(device)

if __name__ == '__main__':
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
    detector = EmotionDetector(fromPath = config.EMOTION_MODEL)
    AffectName = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

    while(cap.isOpened()):
        t = time.time()
        ret, frame = cap.read()

        if ret:
            roiFaces,facialPoints,box_coords = Preprocess(frame)

            emotions, probs = recognizeEmotion(roiFaces)            

            df = _write_row(df, cnt, box_coords, emotions, probs)

            # Draw bounding box for visiualization
            frame = draw_faces_and_emotion(frame, box_coords, emotions)
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