# 使用 tf.keras 和預訓練的 VGG16 模型進行 CIFAR-10 分類
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import subprocess
import os

log_dir = "logs/5-3"
os.makedirs(log_dir, exist_ok=True)  # 確保 iris 目錄存在
os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)  # 確保 train 子目錄存在
# 載入 CIFAR-10 數據集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 預處理數據
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 加載預訓練的 VGG16 模型（不包含頂層）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 凍結 VGG16 的所有層
base_model.trainable = False

# 建立新的模型
model = Model(inputs=base_model.input, outputs=Dense(10, activation='softmax')(Flatten()(base_model.output)))

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型並使用 TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# 可視化 TensorBoard
print(f"TensorBoard logs are stored in {log_dir}")
subprocess.run(["tensorboard", "--logdir", log_dir])
