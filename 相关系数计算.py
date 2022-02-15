import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_excel('D:\桌面\学习相关\大三上\管理统计学\好奇心数据-第二次作业 (1).xlsx',sheet_name='好奇心数据总')
#print(data)
lista = [data['被剥夺感均'],data['快乐探索均'],data['社交好奇均'],data['抗压能力均'],data['寻求刺激均']]

column_lst = ['被剥夺感均', '快乐探索均', '社交好奇均', '抗压能力均', '寻求刺激均']
data_dict = {} # 创建数据字典，为生成Dataframe做准备
for col, gf_lst in zip(column_lst, lista):
    data_dict[col] = gf_lst
unstrtf_df = pd.DataFrame(data_dict)
cor1 = unstrtf_df.corr(method='spearman') # 计算相关系数，得到一个矩阵
#method='spearman'
plt.subplots(figsize=(9, 9)) # 设置画面大小
ax=sns.heatmap(cor1, annot=True, vmax=1,vmin = 0,  cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#sns.heatmap(cor1, annot=True, vmax=1, cmap="Blues")
#plt.savefig('./BluesStateRelation.png')
plt.rcParams['font.sans-serif']=['Simhei']
plt.show()
C=np.linalg.inv(cor1) 
VIF=np.diag(C)
VIF.round(2)
print(VIF)
print(cor1)
print(unstrtf_df.columns.tolist())