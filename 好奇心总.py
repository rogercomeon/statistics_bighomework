import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
#与性别
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='好奇心数据总')
data= data.dropna(axis='index', how='any', subset=['性别1'])
#print(type(data))
features=['好奇心总度量']
labels = ["性别1"]
y=data[labels]
X = sm.add_constant(data[features])
model = sm.Logit(y, X)
result = model.fit()
print(result.summary())
print (np.exp(result.params))

#与专业
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='好奇心数据总')
data= data.dropna(axis='index', how='any', subset=['性别1'])
#print(type(data))
features=['好奇心总度量']
labels = ["专业"]
y=data[labels]
X = sm.add_constant(data[features])
model = sm.Logit(y, X)
result = model.fit()
print(result.summary())
print (np.exp(result.params))