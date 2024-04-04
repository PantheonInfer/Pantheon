from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class GTSRB(Dataset):
    base_folder = 'gtsrb'
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.csv_file_name = 'Train.csv' if train else 'Test.csv'

        csv_file_path = os.path.join(
            self.root_dir, self.base_folder, self.csv_file_name
        )
        self.csv_data = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root_dir, self.base_folder, self.csv_data.loc[idx, 'Path'].lower())
            img = Image.open(img_path)
        except:
            folder, name = self.csv_data.loc[idx, 'Path'].lower().split('/')
            img_path = os.path.join(self.root_dir, self.base_folder, folder, str(self.csv_data.loc[idx, 'ClassId']), name)
            img = Image.open(img_path)
        class_id = self.csv_data.loc[idx, 'ClassId']
        if self.transform is not None:
            img = self.transform(img)

        return img, class_id