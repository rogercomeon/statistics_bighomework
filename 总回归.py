import numpy as np
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='好奇心数据总')
features = ['好奇心总度量']
labels = ["成绩"]
data= data.dropna(axis='index', how='any', subset=['成绩'])
#y=data[['成绩']]
#print(type(y))
#一个[]，则是series类型，两个[]，则是dataframe类型
y=data[labels]
X = sm.add_constant(data[features])
model = sm.OLS(y, X)
result = model.fit()
#print(y)
print(result.summary())