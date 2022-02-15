import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='Sheet4')
#data= data.dropna(axis='index', how='any', subset=['性别1'])
#print(type(data))
features=['被剥夺感均', '快乐探索均', '社交好奇均', '抗压能力均', '寻求刺激均']
labels = ["专业"]
y=data[labels]
print(type(y))
X = sm.add_constant(data[features])
model = sm.Logit(y, X)
result = model.fit()
print(result.summary())
print (np.exp(result.params))