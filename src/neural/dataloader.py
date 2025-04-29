import pandas as pd

class DataLoader:

    def __init__(self):
        pass

    def load_data(self):
        X_train = pd.read_csv("Data/xtrd.csv")
        y_train = pd.read_csv("Data/ytrd.csv")

        X_val = pd.read_csv("Data/xvald.csv")
        y_val = pd.read_csv("Data/yvald.csv")

        X_test = pd.read_csv("Data/xtstd.csv")
        y_test = pd.read_csv("Data/ytstd.csv")

        return X_train, y_train, X_val, y_val, X_test, y_test

        