import subprocess
from pathlib import Path


def get_all_modules(source: Path = "./kliff") -> list[str]:
    """
    Get all modules of the package.

    Note, this only get the first-level modules like `kliff.module_a`, not modules
    (in subpackages) like `kliff.subpackage_a.module_b`. subpackage is considered
    as a module.

    This takes advantage of
        $ sphinx-apidoc -f -e -o <outdir> <sourcedir>
    Return a list of modules names.
    """
    outdir = Path("/tmp/kliff_apidoc")
    subprocess.run(
        ["sphinx-apidoc", "-f", "-e", "-o", outdir, source],
        check=True,
    )

    # every generated file is kliff.<module>.rst
    modules = [
        p.stem.split(".", 1)[1]          # keep text after 'kliff.'
        for p in outdir.glob("kliff.*.rst")
        if p.stem != "kliff"             # skip the package page itself
    ]
    return sorted(set(modules))



def autodoc_package(path: Path, modules: list[str]):
    """
    Create a package reference page.

    Args:
        path: directory to place the file
        modules: list of API modules
    """
    path = Path(path).resolve()
    if not path.exists():
        path.mkdir(parents=True)

    with open(path / "kliff.rst", "w") as f:
        f.write(".. _reference:\n\n")
        f.write("Package Reference\n")
        f.write("=================\n\n")
        f.write(".. toctree::\n")
        for m in modules:
            f.write("    kliff." + m + "\n")


def autodoc_module(path: Path, module: str):
    """
    Create a module reference page.

    Args:
        path: directory to place the file
        module: name of the module
    """
    path = Path(path).resolve()
    if not path.exists():
        path.mkdir(parents=True)

    module_name = "kliff." + module
    fname = path.joinpath(module_name + ".rst")
    with open(fname, "w") as f:
        f.write(f"{module_name}\n")
        f.write("-" * len(module_name) + "\n\n")
        f.write(f".. automodule:: {module_name}\n")
        f.write("    :members:\n")
        f.write("    :undoc-members:\n")
        # f.write("    :show-inheritance:\n")
        f.write("    :inherited-members:\n")


def create_apidoc(directory: Path = "./apidoc"):
    """
    Create API documentation, a separate page for each module.

    Args:
        directory:
    """

    # modules with the below names will not be excluded
    excludes = ["cmdline"]

    package_path = Path(__file__).parents[2] / "kliff"
    modules = get_all_modules(package_path)
    for exc in excludes:
        modules.remove(exc)
    modules = sorted(modules)

    autodoc_package(directory, modules)
    for mod in modules:
        autodoc_module(directory, mod)


if __name__ == "__main__":
    create_apidoc(directory="./source/apidoc")
