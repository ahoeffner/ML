import joblib
import pandas as pd
from numpy import ndarray
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Predictions :
	def __init__(self, path: str) :
		if (path) :
			self.load(path)
			self.prepare()

			X_val, Y_val = self.getValidationData()
			X_train, Y_train = self.getTrainingData()

			print(f"\n\nTraining Data:\n\n{X_train}\n\n")

			lr = LiniearRegression()
			lr.train(X_train, Y_train)
			lr.save("LiniearRegression.model")
			lr.load("LiniearRegression.model")

			pred = lr.predict(X_val)
			self.print("LiniearRegression", Y_val, pred)

			print("\n")

			rf = RandomForest()
			rf.train(X_train, Y_train)
			rf.save("RandomForest.model")
			rf.load("RandomForest.model")

			pred = rf.predict(X_val)
			self.print("RandomForest", Y_val, pred)



	def predict(self,df:pd.DataFrame) :
		self.df = df
		df['hire_date'] = pd.to_datetime(df['hire_date'])
		self.df = self.df.rename(columns={"salary": "SALARY"})
		self.prepare()
		X_test, ign = self.getTrainingData()
		print("Data:")
		print(X_test)


	def prepare(self) :
		today = pd.to_datetime("today")

		self.df["CEO"] = 0
		self.df["MGR"] = 0
		self.df["CMGR"] = 0
		self.df["SNRMGR"] = 0

		self.df["TSC"] = 0
		self.df["CONS"] = 0
		self.df["SNRTSC"] = 0
		self.df["SNRCONS"] = 0

		self.df["TENURE"] = (today - self.df["hire_date"]).dt.days / 365

		for i,row in self.df.iterrows() :
			self.df.at[i,row["job_id"]] = 1

		self.split()


	def diff(self,rv,ev) :
		diff = abs(rv - ev) * 100 / rv
		result = f"{diff:.2f}%"
		if (diff > 5) : result = "* " + result
		return result


	def print(self, model:str, facts:pd.DataFrame, predicted:ndarray) :
		print(f"\n\n{model} Predictions")
		print("-------------------------------------------")
		result = facts.copy()
		result = result.rename(columns={"attrition": "ACTUAL"})
		result["PREDICTED"] = predicted
		result["DIFF"] = result.apply(lambda row: self.diff(row['ACTUAL'],row['PREDICTED']), axis=1)
		print(result)
		print("R2 Score: ", r2_score(facts, predicted), "\n")


	def load(self,path:str) :
		self.df = pd.read_csv(path,parse_dates=["hire_date"])
		self.df = self.df.rename(columns={"salary": "SALARY"})


	def split(self) :
		shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

		tr = int(len(shuffled) * 0.8)
		tst = int((len(shuffled) - tr)/2)

		self.train = shuffled[:tr]

		self.test = shuffled[tr:tr+tst]
		self.validate = shuffled[tr+tst:]

		self.target = ["attrition"]
		self.features = ["CEO","MGR","CMGR","SNRMGR","TSC","CONS","SNRTSC","SNRCONS","TENURE","SALARY"]


	def getTrainingData(self) :
		return self.train[self.features], self.train[self.target]


	def getTestData(self) :
		return self.test[self.features], self.test[self.target]


	def getValidationData(self) :
		return self.validate[self.features], self.validate[self.target]


class RandomForest :
	def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame) :
		self.model = RandomForestRegressor(n_estimators=1000, random_state=42)
		self.model.fit(X_train, Y_train.values.ravel())
		print("RandomForest trained")

	def save(self, path:str) :
		joblib.dump(self.model, path)
		print("RandomForest Model saved to: ",path)

	def load(self, path:str) :
		self.model = joblib.load(path)
		print("RandomForest Model loaded from: ",path)

	def predict(self, df:pd.DataFrame) :
		return self.model.predict(df)



class LiniearRegression :
	def train(self, X_train:pd.DataFrame, Y_train:pd.DataFrame) :
		self.model = LinearRegression()
		self.model.fit(X_train, Y_train)
		print("LiniearRegression fitted")


	def save(self, path:str) :
		joblib.dump(self.model, path)
		print("LiniearRegression Model saved to: ",path)

	def load(self, path:str) :
		self.model = joblib.load(path)
		print("LiniearRegression Model loaded from: ",path)

	def predict(self, df:pd.DataFrame) :
		return self.model.predict(df)



if __name__ == "__main__":
	data = Predictions("/Users/alhof/Repository/GenAI/bronze/attrition.csv")