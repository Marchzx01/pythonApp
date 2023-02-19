import pandas as pd
import joblib as jb
# pd=pd.read_excel("./ussi/LinearRegression.xlsx",index_col=None)
pd=pd.read_excel("house.xlsx",index_col=None)
# pd=pd.read_excel("./ussi/LinearRegression.xlsx",skiprows=range(1,3),usecols=range(4,7),nrows=16,header=0, names=['co', 'co', 'co'] ,index_col=None)
# pd=pd.read_excel('./ussi/LinearRegression.xlsx',header=0 ,usecols="A,D",nrows=11,skiprows=0,index_col=None)
print(pd)

# op=jb.load("model.joblib")
# print(op)