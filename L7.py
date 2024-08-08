import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成数据
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 打乱数据并分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化线性回归模型
lin_reg = LinearRegression()

# 训练模型
lin_reg.fit(X_train_scaled, y_train)

# 使用训练好的模型进行预测
y_train_pred = lin_reg.predict(X_train_scaled)
y_test_pred = lin_reg.predict(X_test_scaled)

# 计算残差
train_residuals = (y_train - y_train_pred).flatten()
test_residuals = (y_test - y_test_pred).flatten()

# 预测误差分布图
plt.figure()
sns.histplot(test_residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Prediction Error Distribution Plot')
plt.show(block=True)  # 显示图1

# 残差的自相关图(基于残差独立且他且同分布的假设)
plt.figure()
sm.graphics.tsa.plot_acf(test_residuals, lags=20)  # 使用20个滞后期
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Residuals Plot')
plt.show(block=True)  # 显示图2

# Cook's 距离图
lin_reg = sm.OLS(y_train, sm.add_constant(X_train_scaled)).fit()  # 使用statsmodels重新拟合模型
influence = lin_reg.get_influence()
(c, p) = influence.cooks_distance
plt.figure()
plt.stem(np.arange(len(c)), c, markerfmt=",")
plt.xlabel('Index')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance Plot")
plt.show(block=True)  # 显示图3

# Leverage vs. Residuals Plot
leverage = influence.hat_matrix_diag
plt.figure()
plt.scatter(leverage, train_residuals)
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.title('Leverage vs. Residuals Plot')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.show(block=True)  # 显示图4

# 分布图
plt.figure()
sns.kdeplot(y_test.flatten(), label='Actual', color='blue')
sns.kdeplot(y_test_pred.flatten(), label='Predicted', color='red')
plt.xlabel('Value')
plt.title('Distribution Plot')
plt.legend()
plt.show(block=True)  # 显示图5

1
2
3