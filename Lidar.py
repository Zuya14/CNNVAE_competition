import torch
import numpy as np
from pathlib import Path

class LidarDatasets(torch.utils.data.Dataset):

    def __init__(self, filenames, maxLen=1.0):
        self.data = np.concatenate([np.load(filename) for filename in filenames])
        self.data = torch.tensor(self.data).float()

        self.datanum = self.data.shape[0]
        self.datasize = self.data.shape[-1]

    def limit(self, end):
        self.data = self.data[:, :end]

        self.datanum = self.data.shape[0]
        self.datasize = self.data.shape[-1]

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':

    train_filenames = ['./data/vaernnEnv0/id-{}.npy'.format(id) for id in range(10)]
    test_filenames  = ['./data/vaernnEnv0/id-{}.npy'.format(id) for id in range(10, 12)]

    lidarTrainDatasets = LidarDatasets(train_filenames)
    lidarTrainDatasets.limit(1080)

    lidarTestDatasets = LidarDatasets(train_filenames)
    lidarTestDatasets.limit(1080)