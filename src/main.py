import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Data :
	def __init__(self, path: str) :
		self.load(path)
		self.prepare()

		X_test, ign = self.getTrainingData()
		print("Data:")
		print(X_test)

		self.liniearRegression()
		self.randomForest()



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

		self.df["tenure"] = (today - self.df["hire_date"]).dt.days / 365

		for i,row in self.df.iterrows() :
			noise = len(self.df.at[i,"first_name"]) + len(self.df.at[i,"last_name"])
			self.df.at[i,row["job_id"]] = 1
			self.df.at[i,"noise"] = 100*noise

		self.split()



	def liniearRegression(self) :
		X_test, y_test = self.getTestData()
		X_train, y_train = self.getTrainingData()

		model = LinearRegression()
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		result = y_test.copy()
		result["predicted"] = y_pred
		result["diff"] = result.apply(lambda row: self.diff(row['attrition'],row['predicted']), axis=1)

		print("\n\nLiniearRegression")
		print("-------------------------------------------")
		print("\n",result)
		r2 = r2_score(y_test, y_pred)
		print("\nR2 Score: ",round(r2,5),"\n")
		print("-------------------------------------------")



	def randomForest(self) :
		X_test, y_test = self.getTestData()
		X_train, y_train = self.getTrainingData()

		model = RandomForestRegressor(n_estimators=1000, random_state=42)
		model.fit(X_train, y_train.values.ravel())
		y_pred = model.predict(X_test)

		result = y_test.copy()
		result["predicted"] = y_pred
		result["diff"] = result.apply(lambda row: self.diff(row['attrition'],row['predicted']), axis=1)

		print("\n\nRandomForest estimators=1000")
		print("-------------------------------------------")
		print("\n",result)
		r2 = r2_score(y_test, y_pred)
		print("\nR2 Score: ",round(r2,5),"\n")
		print("-------------------------------------------")


	def diff(self,rv,ev) :
		diff = abs(rv - ev) * 100 / rv
		result = f"{diff:.2f}%"
		if (diff > 5) : result = "* " + result
		return result



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
		self.features = ["noise","CEO","MGR","CMGR","SNRMGR","TSC","CONS","SNRTSC","SNRCONS","salary","tenure"]


	def getTrainingData(self) :
		return self.train[self.features], self.train[self.target]


	def getTestData(self) :
		return self.test[self.features], self.test[self.target]


	def getValidationData(self) :
		return self.validate[self.features], self.validate[self.target]



data = Data("/Users/alhof/Repository/GenAI/bronze/attrition.csv")