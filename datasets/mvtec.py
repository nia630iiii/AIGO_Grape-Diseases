import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'

#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

CLASS_NAMES = ['grape']
class MVTecDataset(Dataset):
    def __init__(self, dataset_path='', class_name='ripe', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y= self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        
        for img_type in img_types:
            img_type_path = os.path.join(img_dir, img_type)
            x.append(img_type_path)

        return list(x), list(y)
