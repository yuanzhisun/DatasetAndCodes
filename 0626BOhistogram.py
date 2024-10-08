import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 数据
file_path = 'F:/H621/BO-histogram.xlsx'
df = pd.read_excel(file_path)

# 设置图形字体和字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# 创建绘图窗口
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

# 绘制第一个点线图
axs[0].plot(df.iloc[:, 0], df.iloc[:, 1], label='GBDT-BO', color='blue', linewidth=1, marker='o', markersize=3)
axs[0].set_ylabel('RMSE values')
axs[0].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(1)

# 绘制第二个点线图
axs[1].plot(df.iloc[:, 0], df.iloc[:, 2], label='BPNN-BO', color='green', linewidth=1, marker='o', markersize=3)
axs[1].set_ylabel('RMSE values')
axs[1].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(1)

# 绘制第三个点线图
axs[2].plot(df.iloc[:, 0], df.iloc[:, 3], label='SVR-BO', color='red', linewidth=1, marker='o', markersize=3)
axs[2].set_ylabel('RMSE values')
axs[2].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(1)

# 绘制第四个点线图
axs[3].plot(df.iloc[:, 0], df.iloc[:, 4], label='XGBoost-BO', color='purple', linewidth=1, marker='o', markersize=3)
axs[3].set_xlabel('Step')
axs[3].set_ylabel('RMSE values')
axs[3].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12).get_frame().set_linewidth(1)

# 设置图框宽度
for ax in axs:
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

# 显示图形
# plt.show()
plt.savefig('F:/H621/Histogram-BO-All.png', dpi=1000)