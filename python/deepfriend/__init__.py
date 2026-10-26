"""DeepFriend: a study buddy for PyTorch and TensorFlow code.

The package only uses the standard library at import time, so it can be
bundled inside the VS Code extension and imported without pip.
"""

__version__ = "0.1.0"


def watch(name=None, config=None, record=None):
    """Start the training doctor and the automatic experiment log.

    Add ``import deepfriend; deepfriend.watch()`` after importing torch or
    keras. Losses, accuracy, learning rate and batch size are collected
    automatically; nothing ever leaves your computer.
    """
    from .recorder import watch as _watch

    return _watch(name=name, config=config, record=record)


def log(step=None, epoch=None, **metrics):
    """Record extra numbers, e.g. ``deepfriend.log(val_loss=0.3, val_acc=0.9)``."""
    from .recorder import log as _log

    _log(step=step, epoch=epoch, **metrics)


def stop(status="finished"):
    """Finish the current run early (normally done automatically at exit)."""
    from .recorder import stop as _stop

    _stop(status)


def keras_callback():
    """A Keras callback, for ``model.fit(..., callbacks=[deepfriend.keras_callback()])``."""
    from .recorder import keras_callback as _cb

    return _cb()

