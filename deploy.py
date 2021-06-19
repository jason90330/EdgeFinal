import os
import os.path as osp
import models
import torch
import torch.backends.cudnn as cudnn
from models.efficientNet import MyEfficientNet
from retina_face import config

def init_model(net, restore, parallel_reload=True, use_gpu=True):
    """Init models with cuda and weights."""
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

    if torch.cuda.is_available() and use_gpu:
        cudnn.benchmark = True
        net.cuda()
    return net

if __name__ == '__main__':
    savePath = config.EMOTION_MODEL.replace("pth","onnx")
    input_shape = (3,224,224)   #输入数据
    batch_size = 1  #批处理大小
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyEfficientNet()
    model.network.set_swish(memory_efficient=False)
    restoredModel = init_model(net=model, restore=config.EMOTION_MODEL, parallel_reload=True)
    restoredModel.eval()

    input_data_shape = torch.randn(batch_size, *input_shape, device=device)
    torch.onnx.export(restoredModel, input_data_shape, savePath, verbose=True)