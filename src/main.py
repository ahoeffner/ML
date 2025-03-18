import math
import pandas as pd
from sklearn.linear_model import LinearRegression


class Data :
	def __init__(self, path: str) :
		self.load(path)
		self.prepare()
		self.liniearRegression()



	def prepare(self) :
		today = pd.to_datetime("today")
		self.df["tenure"] = (today - self.df["hire_date"]).dt.days / 365

		for i,row in self.df.iterrows() :
			if (row["job_id"] == "CEO") : self.df.at[i,"job"] = 1
			elif (row["job_id"] == "TSC") :	self.df.at[i,"job"] = 2
			elif (row["job_id"] == "MGR") :	self.df.at[i,"job"] = 3
			elif (row["job_id"] == "CMGR") : self.df.at[i,"job"] = 4
			elif (row["job_id"] == "CONS") : self.df.at[i,"job"] = 5
			elif (row["job_id"] == "SNRMGR") :	self.df.at[i,"job"] = 6
			elif (row["job_id"] == "SNRTSC") :	self.df.at[i,"job"] = 7
			elif (row["job_id"] == "SNRCONS") : self.df.at[i,"job"] = 8

			self.split()



	def liniearRegression(self) :
		X_test, y_test = self.getTestData()
		X_train, y_train = self.getTrainingData()

		model = LinearRegression()
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		for i in range(len(y_pred)) :
			print(y_test.iloc[i,0],round(y_pred[i][0],2))



	def load(self,path:str) :
		self.df = pd.read_csv(path,parse_dates=["hire_date"])


	def split(self) :
		shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

		tr = int(len(shuffled) * 0.8)
		tst = int((len(shuffled) - tr)/2)

		self.train = shuffled[:tr]

		self.test = shuffled[tr:tr+tst]
		self.validate = shuffled[tr+tst:]

		self.target = ["attrition"]
		self.features = ["job","salary","tenure"]


	def getTrainingData(self) :
		return self.train[self.features], self.train[self.target]


	def getTestData(self) :
		return self.test[self.features], self.test[self.target]


	def getValidationData(self) :
		return self.validate[self.features], self.validate[self.target]



data = Data("/Users/alhof/Repository/GenAI/bronze/attrition.csv")