import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import subprocess

# 確保 log 目錄存在
log_dir = "logs/5-1"
os.makedirs(log_dir, exist_ok=True)  # 確保 iris 目錄存在
os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)  # 確保 train 子目錄存在

# 載入 Iris 數據集
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2)

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 使用 TensorBoard 記錄訓練過程
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# 訓練模型
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# 啟動 TensorBoard 來查看結果
print(f"TensorBoard logs are stored in {log_dir}")
subprocess.run(["tensorboard", "--logdir", log_dir])
