import torch
from torch.utils.data import Dataset
import torchaudio
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class UrbanSound(Dataset):
    base_folder = 'urbansound'
    duration = 2.95
    hop_length = 512
    window_length = 512
    n_mels = 128
    num_samples = 128
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.csv_file_name = 'UrbanSound8K.csv'

        csv_file_path = os.path.join(self.root_dir, self.base_folder, self.csv_file_name)
        csv_data = pd.read_csv(csv_file_path)

        if train:
            self.csv_data = csv_data[(csv_data['fold'] != 9) & (csv_data['fold'] != 10)]
        else:
            self.csv_data = csv_data[(csv_data['fold'] == 9) | (csv_data['fold'] == 10)].reset_index()

        self.csv_data = self.csv_data.reset_index()
        self.transform = transform

        self.inputs = []
        self.labels = []
        for idx, row in tqdm(self.csv_data.iterrows(), total=len(self.csv_data)):
            audio_path = os.path.join(self.root_dir, self.base_folder, 'fold{}'.format(row['fold']),
                                      row['slice_file_name'])
            class_id = row['classID']
            melspec = self._read(audio_path)
            self.inputs.append(self.transform(melspec))
            self.labels.append(class_id)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def _read(self, path):
        audio, sample_rate = librosa.load(path, duration=self.duration, res_type='kaiser_fast')
        melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=self.hop_length,
                                                 win_length=self.window_length, n_mels=self.n_mels)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        length = melspec.shape[1]
        if length != self.num_samples:
            melspec = librosa.util.fix_length(melspec, size=self.num_samples, axis=1, constant_values=(0, -80.0))
        return melspec


# class UrbanSound(Dataset):
#     base_folder = 'urbansound'
#     duration = 2.95
#     hop_length = 512
#     window_length = 512
#     n_mels = 128
#     num_samples = 128
#     def __init__(self, root_dir, train=True, transform=None):
#         self.root_dir = root_dir
#         self.csv_file_name = 'UrbanSound8K.csv'
#
#         csv_file_path = os.path.join(self.root_dir, self.base_folder, self.csv_file_name)
#         csv_data = pd.read_csv(csv_file_path)
#
#         if train:
#             self.csv_data = csv_data[(csv_data['fold'] != 9) & (csv_data['fold'] != 10)]
#         else:
#             self.csv_data = csv_data[(csv_data['fold'] == 9) | (csv_data['fold'] == 10)].reset_index()
#
#         self.csv_data = self.csv_data.reset_index()
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.csv_data)
#
#     def __getitem__(self, idx):
#         audio_path = os.path.join(self.root_dir, self.base_folder, 'fold{}'.format(self.csv_data.loc[idx, "fold"]),
#                                   self.csv_data.loc[idx, 'slice_file_name'])
#         class_id = self.csv_data.loc[idx, 'classID']
#         audio, sample_rate = librosa.load(audio_path, duration=self.duration, res_type='kaiser_fast')
#         melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=self.hop_length,
#                                                  win_length=self.window_length, n_mels = self.n_mels)
#         melspec = librosa.power_to_db(melspec, ref=np.max)
#         length = melspec.shape[1]
#         if length != self.num_samples:
#             melspec = librosa.util.fix_length(melspec, size=self.num_samples, axis=1, constant_values=(0, -80.0))
#
#         return self.transform(melspec), class_id


if __name__ == '__main__':
    from torchvision import transforms as T
    transform = T.Compose([T.ToTensor()])
    us8k = UrbanSound(r'C:\Users\lxhan2\data', train=True, transform=transform)
    for i in range(len(us8k)):
        # if us8k.__getitem__(i)[0].shape[1] != 128 or us8k.__getitem__(i)[0].shape[2] != 128:
        print(us8k.__getitem__(i)[0].shape)