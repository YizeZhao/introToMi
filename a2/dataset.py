import torch
import torch.utils.data as data


class NMDataset(data.Dataset):

    def __init__(self, X, y):

        #4.1
        self.features = X
        self.labels = y

    def __len__(self):

        return len(self.features)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # 4.1
        feature = self.features[index]
        label = self.labels[index]
        return feature, label