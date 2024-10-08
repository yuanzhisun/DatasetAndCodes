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
from sklearn.neural_network import MLPRegressor


# 加载数据集
# 准备特征数据和目标变量
data=pd.read_excel('F:/H621/Ti-V-CralloysV2.xlsx',sheet_name='Sheet10')

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

nd_ultra=pd.DataFrame(nd)


#划分输入与输出
X = nd_ultra.iloc[:,0:19]
y = nd_ultra.iloc[:,19]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# 定义目标函数
def objective(trial):
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-2),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
        'max_iter': trial.suggest_int('max_iter', 200, 1000)
    }

    bpnn = MLPRegressor(**param, random_state=42)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(bpnn, X_train, y_train, cv=kfold)

    original_ytest = y_pred * (max_valueft - min_valueft) + min_valueft
    original_ytrain = y_train * (max_valueft - min_valueft) + min_valueft

    rmse = np.sqrt(mean_squared_error(original_ytrain, original_ytest))
    return rmse

# 使用Optuna进行超参数优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# 获取最佳参数
best_params = study.best_params
print("Best parameters:", best_params)

# 使用最佳参数训练模型
bpnn_optimized = MLPRegressor(**best_params, random_state=42)
bpnn_optimized.fit(X_train, y_train)

# 在训练集和测试集上进行预测
y_train_pred = bpnn_optimized.predict(X_train)
y_test_pred = bpnn_optimized.predict(X_test)

# 将输出进行逆归一化操作
original_ytestpred = y_test_pred * (max_valueft - min_valueft) + min_valueft
original_ytest = y_test * (max_valueft - min_valueft) + min_valueft
original_ytrain = y_train * (max_valueft - min_valueft) + min_valueft
original_ytrainpred = y_train_pred * (max_valueft - min_valueft) + min_valueft


# 评估模型
mse_train = mean_squared_error(original_ytrain, original_ytrainpred)
mse_test = mean_squared_error(original_ytest, original_ytestpred)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)

# 10折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(bpnn_optimized, X, y, cv=kfold)

original_ytest = y_pred * (max_valueft - min_valueft) + min_valueft
original_y = y * (max_valueft - min_valueft) + min_valueft

# 计算10折的平均RMSE与R2
rmse = np.sqrt(mean_squared_error(original_y, original_ytest))
mae = np.mean(np.abs(np.array(original_y) - np.array(original_ytest)))
r2 = r2_score(original_y, original_ytest)
rmse_avg = np.mean(rmse)
r2_avg = np.mean(r2)
mae_avg = np.mean(mae)

# 设置全局字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 绘制预测值与实际值散点图
colors = [(23, 190, 207)]
normalized_colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colors]

# plt.figure(figsize=(2, 2))  # 设置图框宽度和高度
plt.scatter(original_y, original_ytest, c=normalized_colors, edgecolors='black', alpha=0.8, linewidths=1, marker='o',s=55.5)

# 对角线从图框的左下角连接到右上角
plt.plot([-20, 8], [-20, 8], '--', color='black')

# 设置整个图框的线宽为2磅
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# 显示平均RMSE、R2和MAE的文本框在整个图框的左上角
bbox_props = dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='black')
plt.text(0.05, 0.95,
         f"Average RMSE: {rmse:.2f} \nAverage R2: {r2:.2f} \nAverage MAE: {mae:.2f}",
         bbox=bbox_props, ha='left', va='top', transform=plt.gca().transAxes,
         fontweight='bold')

font = {'weight': 'bold', 'size': 12}
plt.xlabel('Exact values of lnKp', fontdict=font)
plt.ylabel('Predicted values of lnKp', fontdict=font)
plt.title('Performance diagram for BPNN', fontdict=font)

# 设置坐标轴范围
plt.xlim(-20, 8)
plt.ylim(-20, 8)

# 设置坐标轴刻度字体为Times New Roman，并加粗
plt.xticks(fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.yticks(fontname='Times New Roman', fontsize=12, fontweight='bold')

# plt.show()
plt.savefig('F:/H621/0626-performance-BPNN.png', dpi=1000)

# # 手动绘制优化历史图
# trials_df = study.trials_dataframe()
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.spines['top'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
#
# plt.scatter(trials_df['number'], trials_df['value'], color='blue', s=50)  # 增加点的大小
# plt.plot(trials_df['number'], trials_df['value'], color='blue', linestyle='-', linewidth=1, marker='o')
#
# plt.xlabel('Step', fontname='Times New Roman', fontsize=12)
# plt.ylabel('RMSE value', fontname='Times New Roman', fontsize=12)
# plt.show()
#
# # 将每一步的trial及对应的Objective value保存到excel 表格中
# trials_df.to_excel('F:/H621/bpnn-trials_optimization_history.xlsx', index=False)
# print("Optimization history saved to bpnn-trials_optimization_history.xlsx")
#
# # 将结果保存到Excel表格
# results_df = pd.DataFrame({
#     'Train MSE': [mse_train],
#     'Test MSE': [mse_test],
#     'Average RMSE': [rmse_avg],
#     'Average R2': [r2_avg],
#     'Average MAE': [mae_avg]
# })
#
# results_df.to_excel('F:/H621/bpnn-optimization_results.xlsx', index=False)
# print("Results saved to bpnn-optimization_results.xlsx")