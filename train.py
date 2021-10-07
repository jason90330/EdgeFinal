from torchvision import datasets, transforms
# from loss.metrics import ArcFace, CosFace, SphereFace, Am_softmax
import torch

import torch.optim as optim
import torch.nn.functional as F
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate  

from loss.label_smooth import LabelSmoothSoftmaxCE
from util.utils import my_collate_fn
# from loss.customFocal import FocalLoss

# from loss.focal import FocalLoss
# from dataset.customDataFerPSin import customData
from dataset.affectnet import customData
# from dataset.customData import customData
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")

from models.efficientNet import MyEfficientNet
net = MyEfficientNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)
net = net.to(device)

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        # SquarePad(),
        transforms.Resize(cfg.INPUT_SIZE),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, contrast = 0.3, saturation = 0.25, hue = 0.05),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),

    'val': transforms.Compose([
        transforms.Resize(cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

image_datasets = {x: customData(train_csv = cfg.TRAIN_CSV_PATH,
                                valid_csv = cfg.VALID_CSV_PATH,
                                max_num = cfg.MAX_OF_EACH_CLASS,
                                data_transforms=data_transforms,
                                dataset=x) for x in ['train','val']}                            

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.BS, shuffle=True, num_workers=10,
                                   collate_fn=my_collate_fn) for x in ['train', 'val']}

params_to_update = list(net.parameters())

print("Params to learn:")
if cfg.FEATURE_EXTRACT:
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

def main():
    ii = 0

    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")

    # criterion: 標準準則 主要用來計算loss
    criterion = LabelSmoothSoftmaxCE()
    # criterion = FocalLoss(outNum = 8, gamma=2, weight = image_datasets['train'].class_sample_count)
    # netOutFeatureNum = net._fc.out_features
    '''
    head_dict = {'ArcFace': ArcFace(in_features = cfg.NET_OUT_FEATURES, out_features = 8, device_id = cfg.GPU_ID),
            'CosFace': CosFace(in_features = cfg.NET_OUT_FEATURES, out_features = 8, device_id = cfg.GPU_ID),
            'SphereFace': SphereFace(in_features = cfg.NET_OUT_FEATURES, out_features = 8, device_id = cfg.GPU_ID),
            'Am_softmax': Am_softmax(in_features = cfg.NET_OUT_FEATURES, out_features = 8, device_id = cfg.GPU_ID)}
    head = head_dict[cfg.HEAD_NAME]
    '''
    # optimizer
    optimizer = torch.optim.AdamW(params=params_to_update, lr=cfg.LR, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    trainAllAcc = []
    trainAllLoss = []
    validAllAcc = []
    validAllLoss = []
    eps = 0
    
    if not os.path.isdir(cfg.MODEL_PATH + "/txt"):
        os.makedirs(cfg.MODEL_PATH + "/txt")
    if not os.path.isdir(cfg.MODEL_PATH + "/output"):
        os.makedirs(cfg.MODEL_PATH + "/output")

    with open(cfg.MODEL_PATH + cfg.ACC_TXT_PATH, "w") as f:
        with open(cfg.MODEL_PATH + cfg.LOG_TXT_PATH, "w")as f2:
            for epoch in range(cfg.PRE_EPOCH, cfg.EPOCH):
                # scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                # head.train()
                train_sum_loss = 0.0
                correct = 0.0                
                total = 0.0
                trainLossEpNow = 0.0
                trainAccEpNow = 0.0
                eps +=1

                for i, data in enumerate(dataloaders_dict['train'], 0):
                    length = len(dataloaders_dict['train'])
                    iterNow = (i + 1 + epoch * length)
                    # warm up learning rate
                    if iterNow<=cfg.WARM_ITER+1:
                        optimizer.param_groups[0]['lr'] = cfg.WARMUP_LR + (iterNow-1) * (cfg.LR-cfg.WARMUP_LR)/cfg.WARM_ITER
                    input, target = data

                    input, target = input.to(device), target.to(device)
                    # clears wi.grad for every weight wi in the optimizer. 
                    optimizer.zero_grad()
                    # forward propagation
                    output = net(input)

                    # warmup_scheduler.dampen() output=35*1000 target=35*1
                    # thetas = head(output,target)
                    # calculate loss by LabelSmoothSoftmaxCE
                    
                    loss = criterion(output, target)
                    '''=center + focal, center need to change to 2 class(haven't test yet)
                    loss_focal = criterion_focal(thetas, target)
                    loss_center = criterion_center(output, target)
                    loss = loss_center + loss_focal
                    '''
                    # loss = criterion(output, target)
                    # backward propagation, will compute the gradient of loss using wi and record at wi.grad
                    loss.backward()
                    # change the learning rate and parameter in specific epoch
                    optimizer.step()       
                    
                    # record the accuracy and loss
                    train_sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()

                    lr = optimizer.param_groups[0]['lr']       
                     
                    trainLossEpNow = train_sum_loss / (i + 1)
                    # trainAccEpNow = 100. * float(correct) / float(total)
                    trainAccEpNow = 100. * float(correct) / float(total)
                    # print('warmup:%e', warmup_optimizer.param_groups[0]['lr'])
                    print('[epoch:%d, iter:%d] | LR: %e | Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, iterNow, lr, trainLossEpNow,
                             trainAccEpNow))
                    f2.write('%03d  %05d | LR: %e | Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, iterNow, lr, trainLossEpNow, trainAccEpNow))
                    f2.write('\n')
                    f2.flush()
                trainAllLoss.append(trainLossEpNow)
                trainAllAcc.append(trainAccEpNow)
                # torch.save(net, 'output/efficientb4_epoch{}.pkl'.format(epoch))
                # torch.save(head, 'output/efficientb4_head_epoch{}.pkl'.format(epoch))

                # 每訓練完一个 epoch 測試一下準確率
                print("Waiting Test!")
                val_sum_loss = 0
                with torch.no_grad():
                    validLossEpNow = 0.0
                    validEpDiff = 0.0
                    correct = 0
                    total = 0
                    for j, data in enumerate(dataloaders_dict['val'], 0):
                        net.eval()
                        input, target = data
                        input, target = input.to(device), target.to(device)
                        output = net(input)
                        # thetas = head(output,target)
                        loss = criterion(output, target)
                        # loss = criterion(output, target)
                        val_sum_loss += loss.item()
                        validLossEpNow = val_sum_loss / (j + 1)
                        optimizer.zero_grad()

                        # 取得分最高的那个類 (output.data的索引)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).cpu().sum()
                    print('測試分類準確度：%.3f%%' % (100. * float(correct) / float(total)))
                    acc = 100. * float(correct) / float(total)
                    scheduler.step(acc)                                     

                    # 將每次測試結果寫入 acc.txt 文件中
                    if (ii % 1 == 0):
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % (cfg.MODEL_PATH, epoch + 1))
                        # torch.save(head.state_dict(), '%s/net_head_%03d.pth' % (cfg.MODEL_PATH, epoch + 1))
                        # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    validAllLoss.append(validLossEpNow)
                    validAllAcc.append(acc)
                    # 記錄最佳測試分類準確率並寫入 best_acc.txt 文件中
                    if acc > best_acc:
                        f3 = open(cfg.MODEL_PATH + cfg.BEST_TXT_PATH, "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

                N = np.arange(0, eps)
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, trainAllAcc, label = "train_acc")
                plt.plot(N, trainAllLoss, label = "train_loss") 
                plt.plot(N, validAllAcc, label = "valid_acc")    
                plt.plot(N, validAllLoss, label = "valid_loss")
                plt.title("Accuracy and Loss")
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend(loc="lower left")
                plt.savefig(cfg.MODEL_PATH +cfg.PLOT_PATH)
            print("Training Finished, TotalEPOCH=%d" % cfg.EPOCH)

if __name__ == "__main__":
    main()