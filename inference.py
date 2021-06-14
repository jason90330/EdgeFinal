import torch
from torch import nn
from torch.nn import Parameter
import math
import torch.nn.functional as F
import os.path as osp
import models
from misc.utils import init_model, init_random_seed, mkdirs

import time
from openvino.inference_engine import IENetwork, IECore
from torchvision import transforms, utils
from PIL import Image
import cv2
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                    if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.50, easy_margin = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------        
        input = torch.Tensor(input)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # --------------------------- convert label to one-hot ---------------------------
        # torch.cuda.set_device(1)
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine) + ((1.0 - one_hot) * cosine)
        output *= 64

        return output

def inferenceByIR():
    # extr_model_xml = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp32/FeatExtor-3.xml" #指定IR模型檔(*.xml)
    # extr_model_bin = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp32/FeatExtor-3.bin" #指定IR權重檔(*.bin)

    # embd_model_xml = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp32/FeatEmbder-3.xml" #指定IR模型檔(*.xml)
    # embd_model_bin = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp32/FeatEmbder-3.bin" #指定IR權重檔(*.bin)

    extr_model_xml = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp16/FeatExtor-3.xml" #指定IR模型檔(*.xml)
    extr_model_bin = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp16/FeatExtor-3.bin" #指定IR權重檔(*.bin)

    embd_model_xml = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp16/FeatEmbder-3.xml" #指定IR模型檔(*.xml)
    embd_model_bin = "results/CelebA_efficient_b0_arc_focal_pretrain++/ir/fp16/FeatEmbder-3.bin" #指定IR權重檔(*.bin)
    ie = IECore() #建立推論引擎
    # net = cv.dnn.readNetFromModelOptimizer(model_xml, model_bin) # 讀取IR檔

    # read model
    extr_net = IENetwork(model=extr_model_xml, weights=extr_model_bin) #載入模型及權重
    extr_input_blob = next(iter(extr_net.inputs))
    extr_out_blob = next(iter(extr_net.outputs))
    extr_exec_net = ie.load_network(network = extr_net, device_name = "CPU")
    n, c, h, w = extr_net.inputs[extr_input_blob].shape #1, 6, 256, 256
    extr_net.batch_size =1

    embd_net = IENetwork(model=embd_model_xml, weights=embd_model_bin) #載入模型及權重
    embd_input_blob = next(iter(embd_net.inputs))
    embd_out_blob = next(iter(embd_net.outputs))
    embd_exec_net = ie.load_network(network = embd_net, device_name = "CPU")

    Head = ArcFace(in_features=1000, out_features=2, device_id=None)
    head_path = osp.join("results/CelebA_efficient_b0_arc_focal_pretrain++/Head-3.pt")
    checkpoint_head = torch.load(head_path)
    Head.load_state_dict(checkpoint_head)
    Head.eval()

    # read data
    RGBimg = Image.open("datasets/1.jpg").convert('RGB').resize((256,256))
    HSVimg = Image.open("datasets/1.jpg").convert('HSV').resize((256,256))
    transf = transforms.Compose([transforms.ToTensor()])
    rgbImg = transf(RGBimg)
    hsvImg = transf(HSVimg)
    allImg = torch.cat([rgbImg,hsvImg], 0)

    # predict
    time_0 = time.clock()
    feats = extr_exec_net.infer(inputs={extr_input_blob: allImg})
    embd = embd_exec_net.infer(inputs={embd_input_blob: feats[extr_out_blob]})
    theta = Head(embd[embd_out_blob], torch.zeros(1))
    time_1 = time.clock()
    probablity = torch.nn.functional.softmax(theta, dim=1).cpu().detach().numpy().copy()
    score = probablity[:,1:]#np.squeeze(score, 1)
    print(score)
    print("Inference time = {:.4f} sec.".format(time_1 - time_0))

def inferenceByNormal():
    FeatExtmodel = models.create("Eff_FeatExtractor")
    FeatEmbdmodel = models.create("Eff_FeatEmbedder")
    FeatExt_path = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatExtor-3.pt"
    FeatEmbd_path = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatEmbder-3.pt"
    FeatExtor = init_model(net=FeatExtmodel, init_type = None, restore=FeatExt_path, parallel_reload=True)
    FeatEmbder= init_model(net=FeatEmbdmodel, init_type = None, restore=FeatEmbd_path, parallel_reload=True)
    Head = ArcFace(in_features=1000, out_features=2, device_id=None)
    head_path = "results/CelebA_efficient_b0_arc_focal_pretrain++/Head-3.pt"
    checkpoint_head = torch.load(head_path)
    Head.load_state_dict(checkpoint_head)
    FeatExtor.eval()
    FeatEmbder.eval()
    Head.eval()

    # read data
    RGBimg = Image.open("datasets/1.jpg").convert('RGB').resize((256,256))
    HSVimg = Image.open("datasets/1.jpg").convert('HSV').resize((256,256))
    transf = transforms.Compose([transforms.ToTensor()])
    rgbImg = transf(RGBimg)
    hsvImg = transf(HSVimg)
    allImg = torch.cat([rgbImg,hsvImg], 0)
    allImg = allImg.unsqueeze(0)

    # predict
    time_0 = time.clock()
    feat  = FeatExtor(allImg)
    embd  = FeatEmbder(feat)
    theta = Head(embd, torch.zeros(1))
    time_1 = time.clock()
    probablity = torch.nn.functional.softmax(theta, dim=1).cpu().detach().numpy().copy()
    score = probablity[:,1:]#np.squeeze(score, 1)
    print(score)
    print("Inference time = {:.4f} sec.".format(time_1 - time_0))


if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES=None
    inferenceByIR()
    # inferenceByNormal()