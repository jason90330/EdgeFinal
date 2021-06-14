import os
import cv2
import config as cfg
import logging
import matplotlib.pyplot as plt
import numpy as np
import itertools
from imutils import paths
from sklearn.metrics import roc_curve, auc, confusion_matrix
import csv

import sys
sys.path.append('model')
sys.path.append('../')

print(sys.path)

# from eval_kit.detectorFERPSin import EmotionDetector
from eval_kit.detector_affect import EmotionDetector

def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    

def model_eval(actual, pred, modelIdx):
    actual = list(map(lambda el:[el], actual))
    pred = list(map(lambda el:[el], pred))
    cm = confusion_matrix(actual, pred)
    plt.figure()
    labelName = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    plot_confusion_matrix(cm, classes=labelName, normalize=True,
                    title="confusion matrix")
    # plt.savefig(""cfg.CONFUSION_PATH)
    plt.savefig("%s/CM_norm%03d.png" %(cfg.MODEL_PATH, modelIdx))
    
    plt.figure()
    plot_confusion_matrix(cm, classes=labelName, normalize=False,
                    title="confusion matrix")
    plt.savefig("%s/CM_%03d.png" %(cfg.MODEL_PATH, modelIdx))
    
    # plt.show()
    # print(cm)

def readCSV(path):
    img_paths = []
    labels = []
    count = [0 for i in range(8)]
    with open(path, newline='') as csvfile:
        next(csvfile) # Skip first row
        rows = csv.reader(csvfile)
        for row in rows:
            label = int(row[6])
            path = "dataset/Resized_Manually_Annotated_Images/"+row[0]
            if label <= 7 and os.path.isfile(path) and count[label]<4250:
                count[label]+=1
                labels.append(label)
                img_paths.append(path)
                # print(row)
    return img_paths, labels

def load_test_image():
    n=0    
    images = []
    labels = []
    img_paths, img_labels = readCSV(cfg.VALID_CSV_PATH)    
    lenOfValid = len(img_paths)

    for idx, (path, label) in enumerate(zip(img_paths, img_labels)):
        try:
            n += 1           
            img = read_image(path)
            images.append(img)
            labels.append(label)
        except:
            logging.info("Failed to read image: {}".format(idx))
            raise

        if n == cfg.TEST_BS or idx==lenOfValid-1:
            n = 0
            tmpImages = images
            tmpLabels = labels
            images = []
            labels = []
            yield(tmpImages, tmpLabels)

def run_test(detector_class, imageIter, modelIdx):
    preds = []
    acts = []
    total = 0
    correct = 0
    detector = detector_class(modelIdx)
    img_paths, _ = readCSV(cfg.VALID_CSV_PATH)  
    # f_lb = open(cfg.TEST_LABEL_PATH, "r")
    # imgLabels = f_lb.readlines()[cfg.TEST_LABEL_ST_IDX:]
    
    if not os.path.isdir(cfg.MODEL_PATH + "/txt"):
        os.mkdir(cfg.MODEL_PATH + "/txt")
    with open(cfg.MODEL_PATH + cfg.TEST_ERR_IMG[1:],"a+") as f:
        f.write("="*60)
        f.write('\nModel %03d'%(modelIdx))
        for idx, (imgs, labels) in enumerate(imageIter):
            # try:
            probs = detector.predict(imgs)
            probsMax = np.argmax(probs, axis=1)
            for i in range(len(probs)):
                groundT = int(labels[i])
                pred = int(probsMax[i])
                acts.append(groundT)
                preds.append(pred)
                if groundT!=pred:
                    index = idx*cfg.TEST_BS+i
                    # imgLabel = imgLabels[index]
                    # imgLabel = imgLabel[:-1]
                    # info = imgLabel.split(',')
                    # info = np.array(info)
                    # fileName = info[0]
                    imgPath = img_paths[index]
                    if groundT != pred:
                        f.write('Idex:%06d  ,GT:%06d,  PD:%06d,  Path:%s\n'%(index, groundT, pred, imgPath))                    
                    f.flush()                    
            total += len(labels)
        
        correct += sum([1 for i,j in zip(preds,acts) if i==j])
        acc = 100. * float(correct) / float(total)
        model_eval(acts, preds, modelIdx)

        with open(cfg.MODEL_PATH + cfg.SCORE_PATH[1:],"a+") as f_score:
            f_score.write('%03dTesting accuracy %f\n'%(modelIdx, acc))
        # print("Testing accuracy: %06f"%(acc))

if __name__ == '__main__':
    for modelIdx in range(11,cfg.TEST_MODEL_NUM+1):
        imageIter = load_test_image()
        run_test(EmotionDetector, imageIter, modelIdx)