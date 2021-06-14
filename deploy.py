import os
import os.path as osp
import models
import torch
import torch.backends.cudnn as cudnn
from loss.metrics import ArcFace

def init_model(net, restore, init_type, init= True, pretrain=True, parallel_reload=True):
    """Init models with cuda and weights."""
    # init weights of model
    if init and not pretrain:
        init_weights(net, init_type)
    
    # restore model weights
    if restore is not None and os.path.exists(restore):

        # original saved file with DataParallel
        state_dict = torch.load(restore)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if parallel_reload and "module."in k:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        
        net.restored = True
        print("*************Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net

if __name__ == '__main__':
    # modelName = "Eff_FeatExtractor"
    # restorePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatExtor-3.pt"
    # savePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatExtor-3.onnx"
    # input_shape = (6,256,256)   #输入数据

    # modelName = "Eff_FeatEmbedder"
    # restorePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatEmbder-3.pt"
    # savePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/FeatEmbder-3.onnx"
    # input_shape = (32,128,128)   #输入数据

    batch_size = 1  #批处理大小
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    # torch.load('filename.pth').to(device)
    model = models.create(modelName, pretrain=False)
    model.set_swish(memory_efficient=False)
    restoredModel = init_model(net=model, init_type='xavier', restore=restorePath, init= False, pretrain=False, parallel_reload=True)
    restoredModel.eval()
    '''
    restoredModel = ArcFace(in_features=1000, out_features=2, device_id=None)
    restorePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/Head-3.pt"
    savePath = "results/CelebA_efficient_b0_arc_focal_pretrain++/Head-3.onnx"
    input_shape = (1000,)   #输入数据
    checkpointHead = torch.load(restorePath)
    # checkpoint_head = torch.load(head_path, map_location='cuda:1')
    restoredModel.load_state_dict(checkpointHead)
    restoredModel.eval()

    input_data_shape = torch.randn(batch_size, *input_shape, device="cpu")
    # FeatExtor.set_swish(memory_efficient=False)
    torch.onnx.export(restoredModel, input_data_shape, savePath, verbose=True)
