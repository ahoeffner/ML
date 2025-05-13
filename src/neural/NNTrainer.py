import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
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



class NNTrainer():

	def __init__(self):
		pass

	def test(self):
		print("Test")

	def train_regression_model(self, model, train_feature_matrix:pd.DataFrame, train_targets:pd.DataFrame, batch_size = 16, epochs = 600, lr_init = 1e-4, lr_end = 5e-6):
		train_targets = train_targets.values.ravel()
		train_feature_matrix = train_feature_matrix.values

		train_dataset = NNDataset(train_feature_matrix, train_targets)
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

		#Fetch device (GPU if possible)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print("Device = {}".format(device))

		#Simple Mean Squared Error Loss
		criterion = nn.MSELoss()
		#Use Adam Optimizer with little bit of regularization to prevent overfitting
		optimizer = optim.Adam(model.parameters(), lr = lr_init, weight_decay = 1e-5)
		#Use linear learning rate scheduling, decaying learning rate linearly from lr_init to lr_end.
		#Consider using CosineAnnealing with restarts instead.
		scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = 1.0, end_factor = lr_end / lr_init, total_iters = epochs)

		print("Number of epochs = {}".format(epochs))

		model = model.to(device)

		for epoch in range(epochs):

			model.train()
			running_loss = 0.0
			for inputs, targets in train_dataloader:
					optimizer.zero_grad()

					inputs = inputs.to(device)
					targets = targets.to(device)
					outputs = model(inputs)

					loss = criterion(outputs, targets.view(-1, 1))
					loss.backward()
					optimizer.step()

					running_loss += loss.item()

			print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_dataloader)}")

			scheduler.step()



if __name__ == "__main__" :
	trainer = NNTrainer()
	trainer.test()