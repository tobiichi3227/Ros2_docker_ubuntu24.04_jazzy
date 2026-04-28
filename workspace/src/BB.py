import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# 1. 讀資料
# ===============================
df = pd.read_csv("all_data.csv")

# ===============================
# 2. velocity → ±1（你指定規則）
# vx > 0 → 1
# vx <= 0 → -1
# ===============================
def binarize(v):
    return 1 if v > 0 else -1

for col in [3, 4, 5]:
    df.iloc[:, col] = df.iloc[:, col].apply(binarize)

# ===============================
# 3. direction encoding (8-class)
# ===============================
def encode_direction(vx, vy, vz):
    if vx == 1 and vy == 1 and vz == 1:
        return 1
    elif vx == 1 and vy == 1 and vz == -1:
        return 2
    elif vx == 1 and vy == -1 and vz == 1:
        return 3
    elif vx == -1 and vy == 1 and vz == 1:
        return 4
    elif vx == -1 and vy == -1 and vz == 1:
        return 5
    elif vx == -1 and vy == 1 and vz == -1:
        return 6
    elif vx == 1 and vy == -1 and vz == -1:
        return 7
    elif vx == -1 and vy == -1 and vz == -1:
        return 8
    else:
        return 0

df["dir_class"] = df.apply(
    lambda row: encode_direction(row[3], row[4], row[5]),
    axis=1
)

# ===============================
# 4. feature / label
# ===============================
# px py pz + direction class
X = df.iloc[:, :3].values
dir_feat = df["dir_class"].values.reshape(-1, 1)

X = np.hstack([X, dir_feat])

y = df.iloc[:, 6:].values

# ===============================
# 5. train / test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 6. model
# ===============================
model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=10000,
    random_state=42
)

# ===============================
# 7. train
# ===============================
model.fit(X_train, y_train)

# ===============================
# 8. predict
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 9. eval
# ===============================
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# ===============================
# 10. save model
# ===============================
joblib.dump(model, "model.pkl")

print("模型已保存！")