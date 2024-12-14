# 使用 tf.keras 解決手寫辨認問題
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import subprocess
import os

log_dir = "logs/5-2"
os.makedirs(log_dir, exist_ok=True)  # 確保 iris 目錄存在
os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)  # 確保 train 子目錄存在
# 載入 MNIST 數據集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 數據預處理
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 建立 CNN 模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型並使用 TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# 可視化 TensorBoard
print(f"TensorBoard logs are stored in {log_dir}")
subprocess.run(["tensorboard", "--logdir", log_dir])
