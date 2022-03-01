import config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None) -> None:
        super().__init__()
        self.data  = pd.read_csv(path_to_csv) # csv file contains image_files, and labels
        self.images_folder = images_folder
        self.image_files   = os.listdir(images_folder)
        self.train = train
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)
    
    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace("jpeg", "")

        image = np.array(Image.open(os.path.join(self.images_folder, image_file + ".jpeg")))
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        # image_file is important when creating the test data
        return image, label, image_file # returning image_file because of submission file