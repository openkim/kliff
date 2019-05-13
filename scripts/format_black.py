"""Format all .py file in the `kliff` directory using `black`.
This assumes ``black`` is intalled (https://github.com/python/black).

To run:
$ python format_black.py

"""

import os
import subprocess


def format_using_black(path):
    subprocess.call(
        [
            'black',
            '--quiet',
            '--line-length',
            '80',
            '--skip-string-normalization',
            path,
        ]
    )


if __name__ == '__main__':
    path = os.path.abspath('../')
    print(
        'Formating .py files in directory "{}" and its subdirectories using '
        '"black"...'.format(path)
    )
    format_using_black(path)
    print('Formatting done.')
