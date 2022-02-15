import numpy as np
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='好奇心数据总')
features = ['被剥夺感均', '快乐探索均', '社交好奇均', '抗压能力均', '寻求刺激均']
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
print(result.t_test("被剥夺感均=0"))
print(result.t_test("快乐探索均=0"))
print(result.t_test("社交好奇均=0"))
print(result.t_test("抗压能力均=0"))
print(result.t_test("寻求刺激均=0"))
#print(pearsonr(data['被剥夺感均'],data['成绩']))
data3=data[['被剥夺感均', '快乐探索均', '社交好奇均', '抗压能力均', '寻求刺激均','成绩']]
#sns.pairplot(data3)
plt.rcParams['font.sans-serif']=['Simhei']
#plt.savefig("D:\桌面\学习相关\大三上\管理统计学\小组作业\回归\examples.png")
#plt.show()
print(stats.shapiro(result.resid))
results = pd.DataFrame({'index': y['成绩'], # y实际值
                        'resids': result.resid, # 残差
                        'std_resids': result.resid_pearson, # 方差标准化的残差
                        'fitted': result.predict() # y预测值
                        })
# 1. 图表分别显示
## raw residuals vs. fitted
# 残差拟合图：横坐标是拟合值，纵坐标是残差。
residsvfitted = plt.plot(results['fitted'], results['resids'],  'o')
l = plt.axhline(y = 0, color = 'grey', linestyle = 'dashed') # 绘制y=0水平线
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show(residsvfitted)


## q-q plot
# 残差QQ图：用来描述残差是否符合正态分布。
qqplot = sm.qqplot(results['std_resids'], line='s')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')
plt.title('Normal Q-Q')
plt.show(qqplot)

data1 = data[features]
data2 = y
print(data1.corr())
print(data1.corrwith(data2['成绩']))
#data2['成绩'] series 才行

#sns.pairplot(data1)

plt.rcParams['font.sans-serif']=['Simhei']
plt.show()

