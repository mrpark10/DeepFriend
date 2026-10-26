import textwrap

from deepfriend.findargs import find_argument


def src(s):
    return textwrap.dedent(s).lstrip("\n")


def text_at(code, loc):
    line = code.splitlines()[loc["line"]]
    return line[loc["col"]:loc["endCol"]]


def test_lr_keyword():
    code = src("""
        import torch
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
    """)
    [loc] = find_argument(code, "lr")
    assert text_at(code, loc) == "0.01" and loc["value"] == 0.01 and loc["insert"] is False


def test_lr_missing_inserts_with_default():
    code = src("""
        from torch import optim
        opt = optim.Adam(model.parameters())
    """)
    [loc] = find_argument(code, "lr")
    assert loc["insert"] is True and loc["defaultValue"] == 0.001 and loc["name"] == "lr" and loc["hasArgs"] is True
    line = code.splitlines()[loc["line"]]
    assert line[loc["col"]] == ")" and loc["endCol"] == loc["col"]


def test_sgd_positional_lr():
    code = "import torch\nopt = torch.optim.SGD(m.parameters(), 0.1, momentum=0.9)\n"
    [loc] = find_argument(code, "lr")
    assert text_at(code, loc) == "0.1"


def test_keras_learning_rate_and_scientific():
    code = "from tensorflow import keras\nopt = keras.optimizers.Adam(learning_rate=1e-3)\n"
    [loc] = find_argument(code, "lr")
    assert text_at(code, loc) == "1e-3" and loc["value"] == 0.001 and loc["name"] == "learning_rate"


def test_keras_string_optimizer_has_no_lr():
    code = "model.compile(optimizer='adam', loss='mse')\n"
    assert find_argument(code, "lr") == []


def test_batch_size_dataloader_and_fit():
    code = src("""
        from torch.utils.data import DataLoader
        train = DataLoader(ds, batch_size=2, shuffle=True)
        val = DataLoader(vds, batch_size=256)
    """)
    locs = find_argument(code, "batch_size")
    assert [text_at(code, l) for l in locs] == ["2", "256"]
    code2 = "model.fit(x, y, batch_size=16, epochs=3)\n"
    assert text_at(code2, find_argument(code2, "batch_size")[0]) == "16"


def test_batch_size_missing_in_fit_inserts_default():
    code = "model.fit(x, y, epochs=3)\n"
    [loc] = find_argument(code, "batch_size")
    assert loc["insert"] is True and loc["defaultValue"] == 32 and loc["name"] == "batch_size"


def test_lr_from_variable_is_not_editable():
    code = "LR = 0.1\nopt = torch.optim.SGD(m.parameters(), lr=LR)\n"
    [loc] = find_argument(code, "lr")
    assert loc["value"] is None and text_at(code, loc) == "LR"


def test_utf16_columns_after_korean():
    code = '학습률 = 1; opt = torch.optim.Adam(m.parameters(), lr=0.01)\n'
    [loc] = find_argument(code, "lr")
    assert text_at(code, loc) == "0.01"


def test_syntax_error_returns_empty():
    assert find_argument("opt = (", "lr") == []
