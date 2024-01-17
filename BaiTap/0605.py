import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Đọc dữ liệu từ file txt
data = pd.read_csv('data_banknote_authentication.txt', header=None)

# Phân chia dữ liệu thành features (X) và labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi nhãn sang dạng one-hot encoding
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Xây dựng mô hình Sequential
model = Sequential()

# Thêm các lớp Dense
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile mô hình với hàm mất mát 'categorical_crossentropy' và tối ưu hóa 'adam'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))
