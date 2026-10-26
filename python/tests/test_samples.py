import json

import pytest

from deepfriend.recorder import samples


class FakeRun:
    def __init__(self, d):
        self.ok = True
        self.dir = str(d)


def load(d):
    return json.loads((d / "samples.json").read_text())["items"]


def test_grayscale_images_from_lists(tmp_path):
    imgs = [[[[float((x + y + i) % 7) for x in range(28)] for y in range(28)]] for i in range(10)]
    samples.capture(FakeRun(tmp_path), (imgs, list(range(10))))
    items = load(tmp_path)
    assert len(items) == 8
    assert items[0]["shape"] == [28, 28, 1] and len(items[0]["pixels"]) == 28 * 28
    assert items[3]["label"] == 3
    assert 0 <= min(items[0]["pixels"]) and max(items[0]["pixels"]) == 255




def test_color_images_from_torch(tmp_path):
    torch = pytest.importorskip("torch")
    x = torch.rand(4, 3, 64, 64)
    y = torch.tensor([1, 0, 2, 1])
    samples.capture(FakeRun(tmp_path), (x, y))
    items = load(tmp_path)
    assert items[0]["shape"] == [32, 32, 3] and len(items[0]["pixels"]) == 32 * 32 * 3
    assert [i["label"] for i in items] == [1, 0, 2, 1]


def test_one_hot_labels_become_class_ids(tmp_path):
    samples.capture(FakeRun(tmp_path), ([[0.1, 0.2], [0.3, 0.4]], [[0, 1], [1, 0]]))
    assert [i["label"] for i in load(tmp_path)] == [1, 0]


def test_dict_batches(tmp_path):
    samples.capture(FakeRun(tmp_path), {"image": [[1.0, 2.0, 3.0]], "label": [7]})
    assert load(tmp_path)[0]["label"] == 7


def test_garbage_never_raises(tmp_path):
    samples.capture(FakeRun(tmp_path), object())
    samples.capture(FakeRun(tmp_path), None)
