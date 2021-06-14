from multiprocessing import Pool
from imutils import paths
import cv2
import os

imagePaths = list(paths.list_images("Manually_Annotated_Images"))

def operation(img_idx):
    img_path = imagePaths[img_idx]
    save_path = img_path.replace("Manually_Annotated_Images", "Resized_Manually_Annotated_Images")
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)        
    image = cv2.imread(img_path,flags=img_idx%2)
    result = cv2.resize(image,(224,224))
    cv2.imwrite(save_path, result)

if __name__ == '__main__':
    p = Pool(processes=8)
    p.map(operation, range(len(imagePaths)))
    p.close()