import pandas as pd
from sklearn.model_selection import train_test_split


class Data :
	def __init__(self, path: str) :
		self.load(path)
		self.transform()


	def load(self,path:str) :
		self.df = pd.read_csv(path,parse_dates=["hire_date"])


	def transform(self) :
		today = pd.to_datetime("today")
		self.df["employed"] = (today - self.df["hire_date"]).dt.days / 365

		for i,row in self.df.iterrows() :
			if (row["job_id"] == "CEO") : self.df.at[i,"job"] = 1
			elif (row["job_id"] == "TSC") :	self.df.at[i,"job"] = 2
			elif (row["job_id"] == "MGR") :	self.df.at[i,"job"] = 3
			elif (row["job_id"] == "CMGR") : self.df.at[i,"job"] = 4
			elif (row["job_id"] == "CONS") : self.df.at[i,"job"] = 5
			elif (row["job_id"] == "SNRMGR") :	self.df.at[i,"job"] = 6
			elif (row["job_id"] == "SNRTSC") :	self.df.at[i,"job"] = 7
			elif (row["job_id"] == "SNRCONS") : self.df.at[i,"job"] = 8

	def split(self) :
		self.X = self.df.drop("risc",axis=1)
		self.y = self.df["status"]
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

data = Data("/Users/alhof/Repository/GenAI/bronze/attrition.csv")
print(data.df)