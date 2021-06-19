import torch
from torch import nn
from torch.nn import Parameter
import math
import torch.nn.functional as F
import os.path as osp
import models
import time

from retina_face import config
from deploy import init_model
from models.efficientNet import MyEfficientNet
from openvino.inference_engine import IENetwork, IECore
from torchvision import transforms, utils
from PIL import Image
import cv2
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def inferenceByIR():
    model_xml = "model/AFFECTNET_imbalance_10000/FP16/net_011.xml" #指定IR模型檔(*.xml)
    model_bin = "model/AFFECTNET_imbalance_10000/FP16/net_011.bin" #指定IR權重檔(*.bin)
    ie = IECore() #建立推論引擎
    # net = cv.dnn.readNetFromModelOptimizer(model_xml, model_bin) # 讀取IR檔

    # read model
    net = IENetwork(model=model_xml, weights=model_bin) #載入模型及權重
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network = net, device_name = "CPU")
    n, c, h, w = net.inputs[input_blob].shape #1, 3, 224, 224
    net.batch_size =1

    # read data
    RGBimg = Image.open("data/test_img_rgb/test.jpg").convert('RGB').resize((224,224))
    transf = transforms.Compose([transforms.ToTensor()])
    rgbImg = transf(RGBimg)

    # predict
    time_0 = time.clock()
    output = exec_net.infer(inputs={input_blob: rgbImg})[out_blob]
    time_1 = time.clock()
    output = torch.Tensor(output)
    probablity = torch.nn.functional.softmax(output, dim=-1).cpu().detach().numpy().copy()
    score = probablity#np.squeeze(score, 1)
    print(score)
    print("Inference time = {:.4f} sec.".format(time_1 - time_0))

def inferenceByNormal():        
    model = MyEfficientNet()
    model.eval()

    # read data
    RGBimg = Image.open("data/test_img_rgb/test.jpg").convert('RGB').resize((224,224))
    transf = transforms.Compose([transforms.ToTensor()])
    rgbImg = transf(RGBimg)

    # predict
    rgbImg = rgbImg.unsqueeze(0)
    rgbImg = rgbImg.to(memory_format=torch.channels_last)
    time_0 = time.clock()
    with torch.no_grad():
        output = model(rgbImg)
    time_1 = time.clock()
    probablity = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().copy()
    score = probablity[:]#np.squeeze(score, 1)
    print(score)
    print("Inference time = {:.4f} sec.".format(time_1 - time_0))

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=None
    inferenceByIR()
    # inferenceByNormal()