import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


class Data :
	def __init__(self, path: str) :
		self.load(path)
		self.prepare()
		self.liniearRegression()



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
			self.df.at[i,row["job_id"]] = 1
			self.df.at[i,"noise"] = 0*len(self.df.at[i,"first_name"])

		self.split()



	def liniearRegression(self) :
		X_test, y_test = self.getTestData()
		X_train, y_train = self.getTrainingData()

		model = LinearRegression()
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		result = y_test.copy()
		result["predicted"] = y_pred
		result["diff"] = result.apply(lambda row: f"{round(abs(row['attrition'] - row['predicted']) * 100 / row['attrition'], 2):.2f}%", axis=1)

		print("\n",result)

		r2 = r2_score(y_test, y_pred)
		print("\n\nR2 Score: ",round(r2,5),"\n")



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