import os
INPUT_SIZE = 224
# BS = 60
BS = 10
LR = 1e-3 
CLASS_NUM = 8
# MAX_OF_EACH_CLASS = 4250
MAX_OF_EACH_CLASS = 10000
WARM_ITER = 2000
WARMUP_LR = 0.0
PRE_EPOCH = 0
EPOCH = 30
TEST_MODEL_NUM = 11
TEST_BS = 400
SCORE_PATH = "./txt/score.txt"
TEST_ERR_IMG = "./txt/misclassified.txt"

TRAIN_CSV_PATH = "dataset/training.csv"
VALID_CSV_PATH = "dataset/validation.csv"
FEATURE_EXTRACT = False

# MODEL_PATH = "./model/AFFECTNET"
# MODEL_PATH = "./model/AFFECTNET_sm_batch"
MODEL_PATH = "./model/AFFECTNET_imbalance_10000"
# MODEL_PATH = "./model/AFFECTNET_imbalance_15000"
PLOT_PATH = "/output/plot.png"
ACC_TXT_PATH = "/txt/acc.txt"
LOG_TXT_PATH = "/txt/log.txt"
BEST_TXT_PATH = "/txt/best_acc.txt"