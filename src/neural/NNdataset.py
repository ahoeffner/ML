import torch
from torch.utils.data import Dataset

class NNDataset(Dataset):

    def __init__(self, feature_matrix, targets):
        self.feauture_matrix = feature_matrix
        self.targets = targets 
        self.N = self.feauture_matrix.shape[0]
        assert(self.N == len(targets))

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return (torch.tensor(self.feauture_matrix[idx, :]).float(), torch.tensor(self.targets[idx]).float())

    