# 예제 3: Keras(TensorFlow)로 만든 CNN
# - Keras는 채널이 맨 뒤에 와요: (배치, 높이, 너비, 채널). 배치 자리는 None으로 보여요.
# - model.fit에 콜백을 따로 넣지 않아도 deepfriend.watch()가 자동으로 연결돼요.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import deepfriend

deepfriend.watch(name="keras-cnn")
rng = np.random.default_rng(0)

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])


def fake_digits(n):
    y = rng.integers(0, 10, n)
    x = rng.random((n, 28, 28, 1), dtype=np.float32) * 0.3
    for i, label in enumerate(y):
        x[i, 2 + label * 2 : 4 + label * 2, 4:24, 0] += 0.7
    return x, y


x_train, y_train = fake_digits(2000)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=64, verbose=2)
