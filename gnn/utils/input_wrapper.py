import inspect
import json
import types
from functools import wraps
from pathlib import Path, WindowsPath
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from cv2 import imread
from PIL import Image


def _is_single_input(xx: Any) -> bool:
    """Test if input is iterable but not a str or numpy array."""
    return type(xx) not in (list, tuple, types.GeneratorType)


def handle_single_input(preprocess_hook=lambda x: x):  # noqa: ANN001, ANN201
    def decorator(func):  # noqa: ANN001, ANN201
        @wraps(func)  # noqa: ANN201
        def decorated_func(*args: Dict[str, Any], **kwargs: Dict[str, Any]) -> Any:
            input_index = 0
            if inspect.getfullargspec(func).args[0] == "self":
                input_index = 1
            input_ = args[input_index]
            is_single_input = _is_single_input(input_)
            if is_single_input:
                input_ = [input_]
            args = list(args)
            args[input_index] = list(map(preprocess_hook, input_))
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise TypeError(
                    "Perhaps you function does not accept an Iterable as input?"
                ) from e

            # Unpack to a single element if input is single
            if is_single_input:
                [result] = result
            return result
        return decorated_func
    return decorator


def _is(type_: Any) -> bool:
    return lambda xx: isinstance(xx, type_)


def _is_windows_path(xx: str) -> bool:
    try:
        return _is(WindowsPath)(Path(xx))
    except Exception:
        return False


def imread_windows(path: str) -> np.array:
    image = bytearray(open(path, "rb").read())
    image = np.asarray(image, "uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def imread_buffer(buffer_: Any) -> np.ndarray:
    image = np.frombuffer(buffer_, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def read_json_file(filename: str) -> Dict[str, Any]:
    with open(filename, encoding="utf-8-sig") as fp:
        return json.load(fp)


def cast_image_to_array(xx: Any) -> Dict[str, Any]:
    handlers = {
        _is_windows_path: imread_windows,
        _is(str): imread,
        _is(Path): lambda xx: imread(str(xx)),
        _is(bytes): imread_buffer,
        _is(np.ndarray): np.array,
        _is(Image.Image): np.array,
    }
    for condition, handler in handlers.items():
        if condition(xx):
            return handler(xx)
    raise TypeError(f"Unsupported image type {type(xx)}")


def cast_label_to_dict(xx: Any) -> Dict[str, Any]:
    handlers = {
        _is_windows_path: read_json_file,
        _is(str): read_json_file,
        _is(Path): lambda xx: read_json_file(str(xx)),
        _is(dict): dict,
    }
    for condition, handler in handlers.items():
        if condition(xx):
            return handler(xx)
    raise TypeError(f"Unsupported image type {type(xx)}")


def cast_label_to_list(xx: Any) -> Dict[str, Any]:
    handlers = {
        _is_windows_path: read_json_file,
        _is(str): read_json_file,
        _is(Path): lambda xx: read_json_file(str(xx)),
        _is(list): list,
        _is(dict): dict,
    }
    for condition, handler in handlers.items():
        if condition(xx):
            return handler(xx)
    raise TypeError(f"Unsupported image type {type(xx)}")


def cast_pair_sample(xx: Tuple[Any, Any]) -> Tuple[Any, Any]:
    if _is_single_input(xx):
        dummy_image = np.zeros((1, 1, 3))
        return [dummy_image, cast_label_to_dict(xx)]
    else:
        aa, bb = xx
        aa = cast_image_to_array(aa)
        bb = cast_label_to_dict(bb)
        return (aa, bb)
