from collections.abc import Iterable


def length_equal(a, b):
    if isinstance(a, Iterable) and isinstance(b, Iterable):
        if len(a) == len(b):
            return True
        else:
            return False
    # if one is Iterable and the other is not, we treat them as equal since it
    # can be broadcasted
    else:
        return True
