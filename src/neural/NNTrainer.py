import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from NNdataset import NNDataset
from NNModel import NNModel


class NNTrainer():

    def __init__(self):
        pass

    def train_regression_model_with_validation(self, train_feature_matrix, train_targets, val_feature_matrix, val_targets, hidden_size = 64, batch_size = 16, epochs = 600, lr_init = 1e-4, lr_end = 5e-6):
        train_dataset = NNDataset(train_feature_matrix, train_targets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = NNDataset(val_feature_matrix, val_targets)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        D = train_feature_matrix.shape[1]
        assert(D == val_feature_matrix.shape[1])

        #Fetch device (GPU if possible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device = {}".format(device))

        model = NNModel(D, hidden_size, 1)
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

            # Validation
            model.eval()
            val_predictions = []
            val_actuals = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_actuals.extend(targets.cpu().numpy().flatten())

            r2 = r2_score(val_actuals, val_predictions)
            rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))

            print(f"Epoch {epoch+1}/{epochs}, Validation R^2: {r2}, Validation RMSE: {rmse}")

            print("\n")

            scheduler.step()

        return model