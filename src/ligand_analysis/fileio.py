"""Atomic file writes so interrupted runs never leave partial outputs."""

import json
import os
from pathlib import Path
import tempfile


def write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_text_atomic(path, text):
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json(path, value):
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")
