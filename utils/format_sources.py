"""Format all .py and CPP source files using `black` and `clang-format` respectively.

This assumes both `black` (https://github.com/python/black) and `clang-format`
(https://clang.llvm.org/docs/ClangFormat.html) are installed.

To run:
$ python format_sources.py
"""

import os
import subprocess


def get_files(path, extension=[".cpp", ".hpp", ".h"]):
    all_srcs = []
    path = os.path.abspath(path)
    for root, dirs, files in os.walk(path):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in extension:
                all_srcs.append(os.path.join(root, f))
    return all_srcs


def format_py_code(path):
    path = os.path.abspath(path)
    print(
        'Formating .py files in directory "{}" and its subdirectories using '
        '"black"...'.format(path)
    )
    subprocess.call(
        ["black", "--quiet", "--line-length", "90", "--skip-string-normalization", path]
    )
    print("Formatting .py files done.")


def format_cpp_code(path):
    path = os.path.abspath(path)
    print(
        'Formating CPP files in directory "{}" and its subdirectories using '
        '"clang-format"...'.format(path)
    )
    files = get_files(path)
    for f in files:
        subprocess.call(["clang-format", "-style=file", "-i", f])
    print("Formatting CPP files done.")


if __name__ == "__main__":
    format_py_code("../")
    format_cpp_code("../")
