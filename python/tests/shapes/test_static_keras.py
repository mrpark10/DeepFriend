import ast
import textwrap

from deepfriend.shapes.static_keras import analyze_keras


def run(src):
    src = textwrap.dedent(src).lstrip("\n")
    return src, analyze_keras(ast.parse(src), src)


def ln(src, needle, nth=0):
    return [i for i, l in enumerate(src.splitlines()) if needle in l][nth]


def hints(m):
    return {h["line"]: h["shape"] for h in m["hints"]}


MNIST = """
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
"""


def test_mnist_cnn_hints_and_params():
    src, models = run(MNIST)
    assert len(models) == 1
    m = models[0]
    assert m["name"] == "model" and m["kind"] == "keras"
    h = hints(m)
    assert h[ln(src, "layers.Conv2D(32")] == "(None, 26, 26, 32)"
    assert h[ln(src, "layers.MaxPooling2D", 0)] == "(None, 13, 13, 32)"
    assert h[ln(src, "layers.Conv2D(64")] == "(None, 11, 11, 64)"
    assert h[ln(src, "layers.MaxPooling2D", 1)] == "(None, 5, 5, 64)"
    assert h[ln(src, "layers.Flatten")] == "(None, 1600)"
    assert h[ln(src, "layers.Dense(10")] == "(None, 10)"
    assert m["totalParams"] == 320 + 18496 + 16010
    assert [b["name"] for b in m["blocks"]] == ["conv2d", "max_pooling2d", "conv2d_1", "max_pooling2d_1", "flatten", "dropout", "dense"]
    assert m["diagnostics"] == []
    assert m["needsInput"] is False
    assert m["line"] == ln(src, "model = keras.Sequential")


def test_dense_on_4d_keeps_rank():
    src, models = run("""
import keras
from keras import layers
m = keras.Sequential([keras.Input(shape=(8, 8, 3)), layers.Dense(5)])
""")
    assert hints(models[0])[ln(src, "m = keras")] == "(None, 8, 8, 5)"
    assert models[0]["diagnostics"] == []


def test_conv_after_flatten_rank_error():
    src, models = run("""
import keras
from keras import layers
model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Conv2D(8, 3))
model.add(layers.Dense(10))
""")
    m = models[0]
    d = m["diagnostics"]
    assert len(d) == 1 and d[0]["code"] == "keras_rank"
    assert d[0]["line"] == ln(src, "layers.Conv2D")
    assert d[0]["args"]["actual"] == 2
    h = {x["line"]: x for x in m["hints"]}
    assert h[ln(src, "layers.Flatten")]["shape"] == "(None, 784)"
    assert h[ln(src, "layers.Dense")]["shape"] == "?"


def test_add_style_with_input_shape():
    src, models = run("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
net = Sequential()
net.add(Dense(64, activation="relu", input_shape=(20,)))
net.add(Dropout(0.2))
net.add(Dense(1))
""")
    m = models[0]
    assert m["name"] == "net"
    h = hints(m)
    assert h[ln(src, "Dense(64")] == "(None, 64)"
    assert h[ln(src, "Dense(1)")] == "(None, 1)"
    assert m["totalParams"] == 20 * 64 + 64 + 65
    assert m["endLine"] == ln(src, "Dense(1)")


def test_functional_api():
    src, models = run("""
import tensorflow as tf
inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", strides=2)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)
""")
    m = models[0]
    assert m["name"] == "model"
    h = hints(m)
    assert h[ln(src, "Conv2D(16")] == "(None, 16, 16, 16)"
    assert h[ln(src, "GlobalAveragePooling2D")] == "(None, 16)"
    assert h[ln(src, "outputs = ")] == "(None, 10)"
    assert m["totalParams"] == (3 * 3 * 3 * 16 + 16) + (16 * 10 + 10)


def test_needs_input_and_comment():
    src, models = run("""
import keras
from keras import layers
m = keras.Sequential([layers.Dense(4), layers.Dense(2)])
""")
    assert models[0]["needsInput"] is True and models[0]["hints"] == []
    src, models = run("""
import keras
from keras import layers
# df: input (None, 6)
m = keras.Sequential([layers.Dense(4), layers.Dense(2)])
""")
    assert models[0]["needsInput"] is False
    assert hints(models[0])[ln(src, "m = keras")] == "(None, 2)"
    assert models[0]["totalParams"] == 28 + 10


def test_unknown_layer_question():
    src, models = run("""
import keras
from keras import layers
m = keras.Sequential([
    keras.Input(shape=(10, 4)),
    layers.LSTM(8),
    layers.Dense(1),
])
""")
    h = {x["line"]: x for x in models[0]["hints"]}
    assert h[ln(src, "layers.LSTM")]["shape"] == "?"
    assert h[ln(src, "layers.LSTM")]["unknownReason"] == "unsupported_op:LSTM"
    assert h[ln(src, "layers.Dense")]["shape"] == "?"


def test_batchnorm_params_and_pool_same():
    src, models = run("""
import keras
from keras import layers
m = keras.Sequential([
    keras.Input(shape=(15, 15, 8)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2, padding="same"),
    layers.AveragePooling2D(3, strides=1),
])
""")
    h = hints(models[0])
    assert h[ln(src, "MaxPooling2D")] == "(None, 8, 8, 8)"
    assert h[ln(src, "AveragePooling2D")] == "(None, 6, 6, 8)"
    assert models[0]["totalParams"] == 32


def test_nonpositive():
    src, models = run("""
import keras
from keras import layers
m = keras.Sequential([keras.Input(shape=(2, 2, 1)), layers.Conv2D(4, 3)])
""")
    assert [d["code"] for d in models[0]["diagnostics"]] == ["nonpositive"]


def test_torch_file_has_no_keras_models():
    _, models = run("import torch\nx = torch.zeros(3)\n")
    assert models == []
