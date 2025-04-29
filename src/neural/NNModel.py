import torch.nn as nn
import torch

#Simple neural network layer, with a linear layer, batch normalization and Leaky ReLU activation. 
class NNLayer(nn.Module):
    def __init__(self, input_size, output_size, negative_slope = 0.2):
        super(NNLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.activation = nn.LeakyReLU(negative_slope = negative_slope)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()

        self.layer1 = NNLayer(input_size, hidden_size)
        self.layer2 = NNLayer(hidden_size, hidden_size)

        #Output layer, is a simple linear layer, so the output space is unrestricted.
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
    def predict(self, numpy_array):
        self.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        with torch.no_grad():
            tensor_input = torch.from_numpy(numpy_array).float().to(device)
            tensor_predictions = self(tensor_input)
            numpy_predictions = tensor_predictions.cpu().numpy()

        return numpy_predictions
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)