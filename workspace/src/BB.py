
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 1. 读取 CSV
# ===============================
df = pd.read_csv("all_data.csv")

# 所有数值保留两位小数
df = df.round(2)

print("处理后的数据（前5行）：")
print(df.head())

# ===============================
# 2. 特征与标签
# 前6列 = 输入
# 后3列 = 输出
# ===============================
X = df.iloc[:, :6].values
y = df.iloc[:, 6:].values

# ===============================
# 3. 标准化输入特征
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 4. 建立神经网络模型
# ===============================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=5000,
    random_state=42
)

# ===============================
# 5. 训练模型
# ===============================
model.fit(X_scaled, y)

# ===============================
# 6. 训练集预测
# ===============================
y_pred_train = model.predict(X_scaled)

# ===============================
# 7. 整体评估
# ===============================
mse = mean_squared_error(y, y_pred_train)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

print("\n===== 整体评估 =====")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

# ===============================
# 8. 各输出列评估
# ===============================
cols = ["landX", "landY", "landZ"]

for i, col in enumerate(cols):
    mse_i = mean_squared_error(y[:, i], y_pred_train[:, i])
    rmse_i = np.sqrt(mse_i)
    mae_i = mean_absolute_error(y[:, i], y_pred_train[:, i])
    r2_i = r2_score(y[:, i], y_pred_train[:, i])

    print(f"\n===== {col} =====")
    print(f"MSE  : {mse_i:.6f}")
    print(f"RMSE : {rmse_i:.6f}")
    print(f"MAE  : {mae_i:.6f}")
    print(f"R²   : {r2_i:.6f}")

# ===============================
# 9. 预测新样本
# ===============================
new_sample = pd.DataFrame([[
    -0.15, 0.05, -0.12,   # posX posY posZ
    1.73, 1.73, 1.73      # velX velY velZ
]], columns=df.columns[:6])

# 标准化
new_scaled = scaler.transform(new_sample)

# 预测
pred_land = model.predict(new_scaled)

print("\n===== 新样本预测着陆点 =====")
print(f"landX = {pred_land[0,0]:.2f}")
print(f"landY = {pred_land[0,1]:.2f}")
print(f"landZ = {pred_land[0,2]:.2f}")
joblib.dump(model, "mlp_model.pkl")

# 存標準化器（超重要！）
joblib.dump(scaler, "scaler.pkl")

print("模型已保存！")