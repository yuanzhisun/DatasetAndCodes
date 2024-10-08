import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import pdb
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import ternary
import matplotlib.cm as cm


# 加载数据集
# 准备特征数据和目标变量
data=pd.read_excel('F:/H621/Ti-V-CralloysV2.xlsx',sheet_name='Sheet13')

# 创建 MinMaxScaler 对象
scaler = MinMaxScaler(feature_range=(0, 1))
# 将数据进行归一化处理                            #加上了硅铝比酸碱比等



df2=data.iloc[:,1]
ddf2=df2.values
normalized_2 = scaler.fit_transform(ddf2.reshape(-1,1))
min_value2 = np.min(data.iloc[:, 1])
max_value2 = np.max(data.iloc[:, 1])
nd2=pd.DataFrame(normalized_2)

df3=data.iloc[:,2]
ddf3=df3.values
normalized_3 = scaler.fit_transform(ddf3.reshape(-1,1))
min_value3 = np.min(data.iloc[:, 2])
max_value3 = np.max(data.iloc[:, 2])
nd3=pd.DataFrame(normalized_3)

df4=data.iloc[:,3]
ddf4=df4.values
normalized_4 = scaler.fit_transform(ddf4.reshape(-1,1))
min_value4 = np.min(data.iloc[:, 3])
max_value4 = np.max(data.iloc[:, 3])
nd4=pd.DataFrame(normalized_4)

df5=data.iloc[:,4]
ddf5=df5.values
normalized_5 = scaler.fit_transform(ddf5.reshape(-1,1))
min_value5 = np.min(data.iloc[:, 4])
max_value5 = np.max(data.iloc[:, 4])
nd5=pd.DataFrame(normalized_5)

df6=data.iloc[:,5]
ddf6=df6.values
normalized_6 = scaler.fit_transform(ddf6.reshape(-1,1))
min_value6 = np.min(data.iloc[:, 5])
max_value6 = np.max(data.iloc[:, 5])
nd6=pd.DataFrame(normalized_6)

df7=data.iloc[:,6]
ddf7=df7.values
normalized_7 = scaler.fit_transform(ddf7.reshape(-1,1))
min_value7 = np.min(data.iloc[:, 6])
max_value7 = np.max(data.iloc[:, 6])
nd7=pd.DataFrame(normalized_7)

df8=data.iloc[:,7]
ddf8=df8.values
normalized_8 = scaler.fit_transform(ddf8.reshape(-1,1))
min_value8 = np.min(data.iloc[:, 7])
max_value8 = np.max(data.iloc[:, 7])
nd8=pd.DataFrame(normalized_8)

df9=data.iloc[:,8]
ddf9=df9.values
normalized_9 = scaler.fit_transform(ddf9.reshape(-1,1))
min_value9 = np.min(data.iloc[:, 8])
max_value9 = np.max(data.iloc[:, 8])
nd9=pd.DataFrame(normalized_9)

df10=data.iloc[:,9]
ddf10=df10.values
normalized_10 = scaler.fit_transform(ddf10.reshape(-1,1))
min_value10 = np.min(data.iloc[:, 9])
max_value10 = np.max(data.iloc[:, 9])
nd10=pd.DataFrame(normalized_10)

df11=data.iloc[:,10]
ddf11=df11.values
normalized_11 = scaler.fit_transform(ddf11.reshape(-1,1))
min_value11 = np.min(data.iloc[:, 10])
max_value11 = np.max(data.iloc[:, 10])
nd11=pd.DataFrame(normalized_11)

df12=data.iloc[:,11]
ddf12=df12.values
normalized_12 = scaler.fit_transform(ddf12.reshape(-1,1))
min_value12 = np.min(data.iloc[:, 11])
max_value12 = np.max(data.iloc[:, 11])
nd12=pd.DataFrame(normalized_12)

df13=data.iloc[:,12]
ddf13=df13.values
normalized_13 = scaler.fit_transform(ddf13.reshape(-1,1))
min_value13 = np.min(data.iloc[:, 12])
max_value13 = np.max(data.iloc[:, 12])
nd13=pd.DataFrame(normalized_13)

df14=data.iloc[:,13]
ddf14=df14.values
normalized_14 = scaler.fit_transform(ddf14.reshape(-1,1))
min_value14 = np.min(data.iloc[:, 13])
max_value14 = np.max(data.iloc[:, 13])
nd14=pd.DataFrame(normalized_14)

df15=data.iloc[:,14]
ddf15=df15.values
normalized_15 = scaler.fit_transform(ddf15.reshape(-1,1))
min_value15 = np.min(data.iloc[:, 14])
max_value15 = np.max(data.iloc[:, 14])
nd15=pd.DataFrame(normalized_15)

df16=data.iloc[:,15]
ddf16=df16.values
normalized_16 = scaler.fit_transform(ddf16.reshape(-1,1))
min_value16 = np.min(data.iloc[:, 15])
max_value16 = np.max(data.iloc[:, 15])
nd16=pd.DataFrame(normalized_16)

df17=data.iloc[:,16]
ddf17=df17.values
normalized_17 = scaler.fit_transform(ddf17.reshape(-1,1))
min_value17 = np.min(data.iloc[:, 16])
max_value17 = np.max(data.iloc[:, 16])
nd17=pd.DataFrame(normalized_17)

df18=data.iloc[:,17]
ddf18=df18.values
normalized_18 = scaler.fit_transform(ddf18.reshape(-1,1))
min_value18 = np.min(data.iloc[:, 17])
max_value18 = np.max(data.iloc[:, 17])
nd18=pd.DataFrame(normalized_18)

df19=data.iloc[:,18]
ddf19=df19.values
normalized_19 = scaler.fit_transform(ddf19.reshape(-1,1))
min_value19 = np.min(data.iloc[:, 18])
max_value19 = np.max(data.iloc[:, 18])
nd19=pd.DataFrame(normalized_19)

df20=data.iloc[:,19]
ddf20=df20.values
normalized_20 = scaler.fit_transform(ddf20.reshape(-1,1))
min_value20 = np.min(data.iloc[:, 19])
max_value20 = np.max(data.iloc[:, 19])
nd20=pd.DataFrame(normalized_20)


dfft=data.iloc[:,22]
ddfft=dfft.values
normalized_ft = scaler.fit_transform(ddfft.reshape(-1,1))
min_valueft = np.min(data.iloc[:, 22])
max_valueft = np.max(data.iloc[:, 22])
ndft=pd.DataFrame(normalized_ft)

nd = np.column_stack((nd2,nd3,nd4,nd5,nd6,nd7,nd8,nd9,nd10,nd11,nd12,nd13,nd14,nd15,nd16,nd17,nd18,nd19,nd20,ndft))

nd_ultra = pd.DataFrame(nd, columns=['Ti', 'V', 'Cr', 'Al', 'Si', 'C', 'Mo', 'Nb', 'W', 'Ta', 'Zr', 'Hf', 'Co', 'Ni', 'Fe', 'Sn', 'Y', 'Nd', 'T','lnKp'])
#
# # 特征名称
feature_names = ['Ti', 'V', 'Cr', 'Al', 'Si', 'C', 'Mo', 'Nb', 'W', 'Ta', 'Zr', 'Hf', 'Co', 'Ni', 'Fe', 'Sn', 'Y', 'Nd', 'T','lnKp']
# nd_ultra=pd.DataFrame(nd)
# 划分输入与输出



# 划分输入与输出
X = nd_ultra.iloc[:, 0:19]
y = nd_ultra.iloc[:, 19]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Define XGBoost model with specified hyperparameters
params = {
    'colsample_bytree': 0.833118625,
    'learning_rate': 0.010918518,
    'max_depth': 7,
    'reg_alpha': 0.014540528,
    'reg_lambda': 0.109236729,
    'subsample': 0.50701577
}

xgb_model = xgb.XGBRegressor(**params)
# 训练模型
xgb_model.fit(X_train, y_train)

# 获取特征重要性
# feature_importance = xgb_model.feature_importances_
# feature_names = ['Ti', 'V', 'Cr', 'Al', 'Si', 'C', 'Mo', 'Nb', 'W', 'Ta', 'Zr', 'Hf', 'Co', 'Ni', 'Fe', 'Sn', 'Y', 'Nd', 'T']














































import shap
# Set global font properties
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 16,
    'font.weight': 'bold'
})
explainer=shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X)

#sample_index=222
shap.initjs()
#shap.force_plot(explainer.expected_value,shap_values[1,:],X.iloc[1,:],matplotlib=True)




#shap.summary_plot(shap_values, X,show=True)
# Customize the appearance of the plot
#fig, ax = plt.gcf(), plt.gca()
#fig.set_size_inches(10, 6)  # Adjust the figure size



# Define a custom colormap with a gradient of purple shades from bottom to top
#num_bars = len(ax.patches)
#colors = plt.cm.Purples_r(np.linspace(0.6, 0.2, num_bars))  # Change the values (0.6, 0.2) for different shades

# Set the colors for the bars as a gradient of purple from bottom to top
#for i, bar in enumerate(ax.patches):
##    bar.set_facecolor(colors[i])
 #   bar.set_edgecolor('black')
# Set the title and axis labels

#plt.xlabel("SHAP Value")

# Show the plot
#plt.show()
#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/MeanSHAPALLcharacteristics.png', dpi=1000)



#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/shapplot-WithExtra.png', dpi=1000)

shap.summary_plot(shap_values, X, show=False, max_display=8)cc
plt.xticks(fontsize=16, color='black', fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
#plt.savefig('/Users/yuanzhisun/Desktop/NICE/NoP2O5/shapplot-WithExtra.png', dpi=1000)
# 设置 colorbar 字体
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)
cbar.yaxis.label.set_size(18)
cbar.yaxis.label.set_weight('bold')
plt.xlabel(r"SHAP value (Impact on model output)", fontsize=16, fontweight='bold')
# plt.show()
plt.savefig('F:/H621/0629-SHAP-XGBoost.png', dpi=1000)