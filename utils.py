import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import cv2
import random
import re

class DepData(Dataset):
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)
        self.file_list = [file for file in self.file_list if file.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        label_demo = self.file_list[idx][0]
        if label_demo == 'D':
            label = 0
        else:
            label = 1

        data = []

        for i in range(7):
            img_path = os.path.join(img_name)

            split_list = self.data_dir.split('\\')
            a = split_list[-1]

            if os.path.exists(img_path):
                emo_pic = cv2.imread(img_path, 1)
                emo_pic = emo_pic.transpose((2, 0, 1))
            else:
                emo_pic = np.zeros((3, 224, 224))
            rand_del = random.random()
            if rand_del <= 0.2 and a=='train':
                emo_pic = np.zeros(emo_pic.shape)
            data.append(torch.from_numpy(emo_pic))
        data = torch.stack(data)

        if self.transform:
            image = self.transform(image)

        return data, label
