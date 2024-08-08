#线性回归（监督学习）<根据特征（考虑不同特征的影响程度）打标签>
#拟合的平面：h(x)=a0(偏置项)+a1(权重项)x1+a2(权重项)x2，整合成矩阵
#误差=真实值-预测值 ，误差是独立(拿到样本后先洗牌)且具有相同分布，并且服从均值为0方差为(theta的平方)的高斯分布；似然函数(累乘)->对数似然->最小二乘法


##线性回归代码示例
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成数据
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 绘制生成的数据
plt.figure()
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
lin_reg = LinearRegression()

# 训练模型
lin_reg.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# 计算均方误差和R²得分
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")

# 绘制训练集的回归线
plt.figure()
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data with Regression Line')
plt.legend()
plt.show()

# 绘制测试集的预测结果
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test Data with Regression Line')
plt.legend()
plt.show()


##数据打乱、特征缩放、交叉验证
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

# 使用交叉验证评估模型性能
scores = cross_val_score(lin_reg, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
print(f"Cross-Validation MSE: {-scores.mean()}")

# 训练模型
lin_reg.fit(X_train_scaled, y_train)

# 使用训练好的模型进行预测
y_train_pred = lin_reg.predict(X_train_scaled)
y_test_pred = lin_reg.predict(X_test_scaled)

# 计算均方误差和R²得分
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")

# 绘制训练集的回归线
plt.figure()
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data with Regression Line')
plt.legend()
plt.show()

# 绘制测试集的预测结果
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test Data with Regression Line')
plt.legend()
plt.show()


##残差图、实际与预测对比图、QQ图、学习曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

# 使用交叉验证评估模型性能
scores = cross_val_score(lin_reg, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
print(f"Cross-Validation MSE: {-scores.mean()}")

# 训练模型
lin_reg.fit(X_train_scaled, y_train)

# 使用训练好的模型进行预测
y_train_pred = lin_reg.predict(X_train_scaled)
y_test_pred = lin_reg.predict(X_test_scaled)

# 计算均方误差和R²得分
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")

# 绘制训练集的回归线
plt.figure()
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data with Regression Line')
plt.legend()
plt.show()

# 绘制测试集的预测结果
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test Data with Regression Line')
plt.legend()
plt.show()

# 残差图
plt.figure()
plt.scatter(y_train_pred, y_train_pred - y_train, color='blue', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, color='green', label='Test data')
plt.hlines(y=0, xmin=min(y_train_pred), xmax=max(y_train_pred), color='red')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
plt.show()

# 实际值与预测值对比图
plt.figure()
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted')
plt.show()

# QQ图
import scipy.stats as stats
plt.figure()
stats.probplot(y_test_pred - y_test.flatten(), dist="norm", plot=plt)
plt.title('QQ Plot')
plt.show()

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(lin_reg, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training error')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation error')
plt.xlabel('Training size')
plt.ylabel('MSE')
plt.title('Learning Curve')
plt.legend()
plt.show()

##图形升级
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
