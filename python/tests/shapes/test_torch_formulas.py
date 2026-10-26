import pytest

from deepfriend.shapes.dims import ShapeError, broadcast, fmt_shape, numel
from deepfriend.shapes import torch_formulas as tf


def err(fn, *a, **k):
    with pytest.raises(ShapeError) as ei:
        fn(*a, **k)
    return ei.value


# ---- dims -----------------------------------------------------------------

def test_fmt_shape():
    assert fmt_shape((32, 64, 112, 112)) == "(32, 64, 112, 112)"
    assert fmt_shape((24,)) == "(24,)"
    assert fmt_shape(()) == "()"
    assert fmt_shape(("B", 10)) == "(B, 10)"
    assert fmt_shape(None) == "?"


def test_numel():
    assert numel((2, 3, 4)) == 24
    assert numel(("B", 3)) is None
    assert numel(None) is None


def test_broadcast():
    assert broadcast((4, 1, 3), (5, 1)) == (4, 5, 3)
    assert broadcast(("B", 10), (10,)) == ("B", 10)
    assert err(broadcast, (3,), (4,)).code == "broadcast"


# ---- conv -------------------------------------------------------------------

def test_conv_spec_example():
    assert tf.conv((32, 3, 224, 224), nd=2, in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3) == (32, 64, 112, 112)


def test_conv_tuple_params_and_dilation():
    assert tf.conv((1, 4, 20, 30), nd=2, in_channels=4, out_channels=8, kernel_size=(3, 5), stride=(1, 2), padding=(1, 0), dilation=(2, 1)) == (1, 8, 18, 13)


def test_conv_same_and_valid():
    assert tf.conv((2, 3, 17, 17), nd=2, in_channels=3, out_channels=5, kernel_size=4, padding="same") == (2, 5, 17, 17)
    assert tf.conv((2, 3, 17, 17), nd=2, in_channels=3, out_channels=5, kernel_size=4, padding="valid") == (2, 5, 14, 14)
    assert err(tf.conv, (2, 3, 17, 17), nd=2, in_channels=3, out_channels=5, kernel_size=3, stride=2, padding="same").code == "invalid_arg"


def test_conv_unbatched():
    assert tf.conv((3, 32, 32), nd=2, in_channels=3, out_channels=6, kernel_size=5) == (6, 28, 28)


def test_conv1d():
    assert tf.conv((8, 16, 100), nd=1, in_channels=16, out_channels=32, kernel_size=3, stride=2) == (8, 32, 49)


def test_conv_in_channels_mismatch():
    e = err(tf.conv, (32, 1, 28, 28), nd=2, in_channels=3, out_channels=16, kernel_size=3)
    assert e.code == "conv_in" and e.info == {"expected": 3, "actual": 1}


def test_conv_groups():
    assert tf.conv((1, 8, 10, 10), nd=2, in_channels=8, out_channels=16, kernel_size=3, groups=4) == (1, 16, 8, 8)
    assert err(tf.conv, (1, 8, 10, 10), nd=2, in_channels=8, out_channels=6, kernel_size=3, groups=4).code == "invalid_arg"


def test_conv_too_small():
    assert err(tf.conv, (1, 1, 3, 3), nd=2, in_channels=1, out_channels=1, kernel_size=5).code == "nonpositive"


def test_conv_rank():
    e = err(tf.conv, (32, 784), nd=2, in_channels=1, out_channels=8, kernel_size=3)
    assert e.code == "rank" and e.info["actual"] == 2


def test_conv_symbolic_batch():
    assert tf.conv(("B", 3, 32, 32), nd=2, in_channels=3, out_channels=16, kernel_size=3) == ("B", 16, 30, 30)


def test_conv_symbolic_spatial_unknown():
    assert tf.conv(("B", 3, "H", 32), nd=2, in_channels=3, out_channels=16, kernel_size=3) is None


# ---- pool -------------------------------------------------------------------

def test_maxpool_default_stride():
    assert tf.pool((32, 64, 112, 112), nd=2, kind="max", kernel_size=2) == (32, 64, 56, 56)


def test_maxpool_ceil_mode():
    assert tf.pool((1, 1, 6, 6), nd=2, kind="max", kernel_size=3, stride=2, ceil_mode=True) == (1, 1, 3, 3)
    assert tf.pool((1, 1, 5, 5), nd=2, kind="max", kernel_size=3, stride=2, padding=1, ceil_mode=True) == (1, 1, 3, 3)
    assert tf.pool((1, 1, 6, 6), nd=2, kind="max", kernel_size=3, stride=2, ceil_mode=False) == (1, 1, 2, 2)


def test_pool_padding_too_big():
    assert err(tf.pool, (1, 1, 8, 8), nd=2, kind="max", kernel_size=2, padding=2).code == "invalid_arg"


def test_avgpool():
    assert tf.pool((4, 3, 15, 15), nd=2, kind="avg", kernel_size=3, stride=3) == (4, 3, 5, 5)


def test_adaptive_pool():
    assert tf.adaptive_pool((8, 512, 7, 9), nd=2, output_size=1) == (8, 512, 1, 1)
    assert tf.adaptive_pool((8, 512, 7, 9), nd=2, output_size=(None, 3)) == (8, 512, 7, 3)


# ---- linear / norm / misc ---------------------------------------------------

def test_linear_ok_and_mismatch():
    assert tf.linear((32, 12544), in_features=12544, out_features=10) == (32, 10)
    e = err(tf.linear, (32, 12544), in_features=10000, out_features=10)
    assert e.code == "linear_in" and e.info == {"expected": 10000, "actual": 12544}


def test_linear_3d():
    assert tf.linear((2, 7, 16), in_features=16, out_features=4) == (2, 7, 4)


def test_batchnorm():
    assert tf.batchnorm((8, 16, 5, 5), nd=2, num_features=16) == (8, 16, 5, 5)
    e = err(tf.batchnorm, (8, 16, 5, 5), nd=2, num_features=32)
    assert e.code == "bn_features" and e.info == {"expected": 32, "actual": 16}
    assert err(tf.batchnorm, (8, 16), nd=2, num_features=16).code == "rank"


def test_layernorm():
    assert tf.layernorm((2, 5, 8), normalized_shape=8) == (2, 5, 8)
    assert err(tf.layernorm, (2, 5, 8), normalized_shape=(4, 8)).code == "invalid_arg"


def test_embedding():
    assert tf.embedding((4, 12), num_embeddings=100, embedding_dim=32) == (4, 12, 32)


def test_flatten():
    assert tf.flatten((32, 64, 7, 7)) == (32, 3136)
    assert tf.flatten((32, 64, 7, 7), start_dim=0) == (100352,)
    assert tf.flatten((2, 3, 4, 5), start_dim=1, end_dim=2) == (2, 12, 5)
    assert tf.flatten(("B", 64, 7, 7)) == ("B", 3136)


def test_reshape():
    assert tf.reshape((2, 3, 4), (-1,)) == (24,)
    assert tf.reshape((32, 64, 7, 7), (32, -1)) == (32, 3136)
    e = err(tf.reshape, (2, 3, 4), (5, -1))
    assert e.code == "reshape_numel" and e.info == {"from": 24, "to": "(5, -1)"}
    assert err(tf.reshape, (2, 3, 4), (5, 5)).code == "reshape_numel"


def test_reshape_symbolic_batch():
    assert tf.reshape(("B", 64, 7, 7), ("B", -1)) == ("B", 3136)
    assert tf.reshape(("B", 64, 7, 7), (-1, 3136)) == ("B", 3136)


def test_cat_and_stack():
    assert tf.cat([(2, 3), (2, 5)], dim=1) == (2, 8)
    e = err(tf.cat, [(2, 3), (4, 5)], dim=1)
    assert e.code == "cat_mismatch" and e.info["dim"] == 0
    assert tf.stack([(2, 3), (2, 3)], dim=0) == (2, 2, 3)
    assert err(tf.stack, [(2, 3), (2, 4)], dim=0).code == "cat_mismatch"


def test_squeeze_unsqueeze_permute_transpose_reduce():
    assert tf.squeeze((1, 3, 1, 5)) == (3, 5)
    assert tf.squeeze((1, 3, 1, 5), dim=2) == (1, 3, 5)
    assert tf.squeeze((1, 3, 1, 5), dim=1) == (1, 3, 1, 5)
    assert tf.unsqueeze((3, 5), dim=0) == (1, 3, 5)
    assert tf.unsqueeze((3, 5), dim=-1) == (3, 5, 1)
    assert tf.permute((2, 3, 4), (2, 0, 1)) == (4, 2, 3)
    assert err(tf.permute, (2, 3, 4), (0, 1)).code == "invalid_arg"
    assert tf.transpose((2, 3, 4), 0, 2) == (4, 3, 2)
    assert tf.reduce((2, 3, 4), dim=1) == (2, 4)
    assert tf.reduce((2, 3, 4), dim=(0, 2), keepdim=True) == (1, 3, 1)
    assert tf.reduce((2, 3, 4), dim=None) == ()
