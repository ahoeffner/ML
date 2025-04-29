import joblib
import pandas as pd
from numpy import ndarray
from Plotter import Plotter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Predictions :
	RFMODEL = "RandomForest.model"
	LRMODEL = "LiniearRegression.model"


	def train(self, path:str = None, data:pd.DataFrame = None) :
		self.df = None

		if (path is not None) :
			self.load(path)

		elif (data is not None) :
			df:pd.DataFrame = data
			self.df = self.rename(df)
			self.df['HIRE_DATE'] = pd.to_datetime(self.df['HIRE_DATE'])

		else :
			print("Invalid data type")
			return


		self.prepare()
		X_train, Y_train = self.getTrainingData()

		print(f"\n\nTraining Data:\n\n{X_train}\n\n")

		self.lr = LiniearRegression()
		self.lr.train(X_train, Y_train)

		self.rf = RandomForest()
		self.rf.train(X_train, Y_train)

		print("Models trained")



	def validate(self,model:str) :
		X_val, Y_val = self.getValidationData()

		if (model == "LiniearRegression") :
			pred = self.lr.predict(X_val)
			self.print("LiniearRegression", Y_val, pred)

		elif (model == "RandomForest") :
			pred = self.rf.predict(X_val)
			self.print("RandomForest", Y_val, pred)



	def rename(self, df:pd.DataFrame) :
		df = df.rename(columns={"salary": "SALARY", "hire_date": "HIRE_DATE"})
		return df



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

		self.rename(self.df)
		self.df["TENURE"] = (today - self.df["HIRE_DATE"]).dt.days / 365

		for i,row in self.df.iterrows() :
			self.df.at[i,row["job_id"]] = 1

		self.split()



	def diff(self,rv,ev) :
		diff = abs(rv - ev) * 100 / rv
		result = f"{diff:.2f}%"
		if (diff > 5) : result = "* " + result
		return result



	def print(self, model:str, facts:pd.DataFrame, predicted:ndarray) :
		plotter = Plotter()
		plotter.scatter_plot(predicted, facts, "Actuals vs Predicted", "Actual", "Predicted")
		return
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
		self.df = self.df.rename(columns={"salary": "SALARY", "hire_date": "HIRE_DATE"})


	def split(self) :
		shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

		tr = int(len(shuffled) * 0.8)
		tst = int((len(shuffled) - tr)/2)

		self.data = shuffled[:tr]
		self.test = shuffled[tr:tr+tst]
		self.fact = shuffled[tr+tst:]

		self.target = ["attrition"]
		self.features = ["CEO","MGR","CMGR","SNRMGR","TSC","CONS","SNRTSC","SNRCONS","TENURE","SALARY"]



	def getTrainingData(self) :
		return self.data[self.features], self.data[self.target]


	def getTestData(self) :
		return self.test[self.features], self.test[self.target]


	def getValidationData(self) :
		return self.fact[self.features], self.fact[self.target]




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



if __name__ == "__main__" :
	path = "/Users/alhof/Repository/GenAI/bronze/attrition.csv"

	predictions = Predictions()
	#predictions.train(path=path)

	df = pd.read_csv(path)
	#df = df.rename(columns={"salary": "SALARY"})

	predictions.train(data=df)
	predictions.validate()
