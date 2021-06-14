from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image 
from imutils import paths
import random
import csv
import os

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def readCSV(path, max_num):
    img_paths = []
    labels = []
    count = [0 for i in range(8)]
    with open(path, newline='') as csvfile:
        next(csvfile) # Skip first row
        rows = csv.reader(csvfile)
        for row in rows:
            label = int(row[6])
            path = "dataset/Resized_Manually_Annotated_Images/"+row[0]
            if label <= 7 and os.path.isfile(path) and count[label]<max_num:
                count[label]+=1
                labels.append(label)
                img_paths.append(path)
                # print(row)
    return img_paths, labels

class customData(Dataset):
    def __init__(self, train_csv, valid_csv, dataset = '', max_num=4250, data_transforms=None, loader = default_loader):        
        self.data_transforms = data_transforms
        self.img_path = []
        self.img_label = []
        self.dataset = dataset
        self.loader = loader
        if dataset == 'train':
            self.img_path, self.img_label = readCSV(train_csv, max_num)

        elif dataset == 'val':
            self.img_path, self.img_label = readCSV(valid_csv, max_num)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        try:
            path = self.img_path[item]
            label = self.img_label[item]
            img = self.loader(path)

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms[self.dataset](img)
                except:
                    print("Cannot transform image: {}".format(path))
            return img, label
        except:
            print(item)