"""Create a separate page for each module.

To run:
    $ python genereate_apidoc.py
which generates a directory named `tmp_apidoc`. Do a comparsion of this file

You can provide an `exclude` list.

"""
import os
import subprocess


def get_all_modules(source='../kliff'):
    """Get all modules.
    Note, this only get the first-level modules like `kliff.module_a`, not modules
    (in subpackages) like `kliff.subpackage_a.module_b`. subpackage is considered
    as a module.

    Take advantage of
        $ sphinx-apidoc -f -e -o <outdir> <sourcedir>
    Return a list of modules names.
    """
    results = subprocess.check_output(
        ['sphinx-apidoc', '-f', '-e', '-o', '/tmp/kliff_apidoc', source],
        universal_newlines=True,
    )
    results = results.split('\n')
    modules = []
    for line in results:
        if 'Creating' in line:
            name = line.rstrip('.').split('/')[-1].split('.')
            if len(name) >= 3:
                mod = name[1]
                if mod not in modules:
                    modules.append(mod)
    return modules


def autodoc_package(path, modules):
    if path and not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, 'kliff.rst')
    with open(fname, 'w') as fout:
        fout.write('.. _reference:\n\n')
        fout.write('Package Reference\n')
        fout.write('=================\n\n')
        fout.write('.. toctree::\n')
        for mod in modules:
            fout.write('    kliff.' + mod + '\n')


def autodoc_module(path, module):
    if path and not os.path.exists(path):
        os.makedirs(path)
    module_name = 'kliff.' + module
    fname = os.path.join(path, module_name + '.rst')
    with open(fname, 'w') as fout:
        fout.write('{}\n'.format(module_name))
        fout.write('-' * len(module_name) + '\n\n')
        fout.write('.. automodule:: {}\n'.format(module_name))
        fout.write('    :members:\n')
        fout.write('    :undoc-members:\n')
        fout.write('    :show-inheritance:\n')
        fout.write('    :inherited-members:\n')


if __name__ == '__main__':

    # this will not be included
    excludes = ['scripts']

    modules = get_all_modules()
    for exc in excludes:
        modules.remove(exc)
    modules = sorted(modules)
    path = './tmp_apidoc'
    autodoc_package(path, modules)
    for mod in modules:
        autodoc_module(path, mod)
