import sys

import numpy as np


def _clean(value):
    """Convert numpy numeric types to their python equivalents."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'S':
            return np.char.decode(value).tolist()
        else:
            return value.tolist()
    elif type(value).__module__ == np.__name__:
        conversion = value.item()
        if sys.version_info.major == 3 and isinstance(conversion, bytes):
            conversion = conversion.decode()
        return conversion
    elif sys.version_info.major == 3 and isinstance(value, bytes):
        return value.decode()
    else:
        return value


def _clean_attrs(attrs):
    return {_clean(k): _clean(v) for k, v in attrs.items()}

