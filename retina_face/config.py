# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

# Emotion
RUNING_FPS = 4
# VPATH = "data/video_ir/0603_laught.mp4"
# VPATH = "data/video_ir/0603_disgust.mp4"
# VPATH = "data/video_ir/ir_0312_out_20210603_celeb_ir_baseline_from_0602_60k_baseline.mp4"
# VPATH = "data/video_first_order/FO_jiunda_happy_absolute_2.mp4"
# VPATH = "data/video_rgb/Phone/Will_crop.mov"
# VPATH = "data/video_rgb/Phone/Cathy_crop.mov"
VPATH = "data/video_rgb/V8_rgb/00590Will_Crop.mp4"
# VPATH = "data/video_rgb/Phone/Rebecca_crop.mov"
GUI_MODE = True
CPU = False
RESIZE_SCALE = 0.5

# Emotion model
# EMOTION_MODEL = "model/AFFECTNET_sm_batch/net_011.pth"
EMOTION_MODEL = "model/AFFECTNET_imbalance_10000/net_011.pth"