import joblib
import pandas as pd
from NNModel import NNModel
from NNTrainer import NNTrainer


class Neural:
	def __init__(self, input_size:int, hidden_size:int, output_size:int) :
		self.model = NNModel(input_size, hidden_size, output_size)
		self.trainer:NNTrainer = NNTrainer()


	def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame) :
		self.trainer.train_regression_model(self.model, X_train, Y_train)
		pass


	def save(self, path:str) :
		joblib.dump(self.model, path)
		print("NeuralNetwork Model saved to: ",path)


	def load(self, path:str) :
		self.model = joblib.load(path)
		print("NeuralNetwork Model loaded from: ",path)

	def predict(self, df:pd.DataFrame) :
		return(self.model.predict(df.values))
