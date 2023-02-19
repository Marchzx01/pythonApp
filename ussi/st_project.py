import pandas as pd
list=[]
ex = pd.read_excel("pop2540.xls",skiprows=range(1,4),usecols=range(4,7),nrows=16)
for i in range(2,20):
    for_pd=pd.read_excel("pop2540.xls",skiprows=range(1,i),usecols=range(4,7),nrows=1, header=0, names=['co', 'co', 'co'] ,index_col=None)
    # s = for_pd.rename(columns={0: 'col1', 1: 'col2', 2: 'col3', })
    list.append(for_pd.values.tolist())
print(list)
pd.DataFrame(list)
print(list)
# print(ex)