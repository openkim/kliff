from pathlib import Path


def update_version(version, path, key, in_quotes=False, extra_space=False):
    """
    Update version given in `key=version` structure.

    Args:
        version: new version to update
        path: path to the file
        key: identifier to search the line, e.g. `__version__`, `version =`.
    """
    with open(path, "r") as fin:
        lines = fin.readlines()
    with open(path, "w") as fout:
        for line in lines:
            if key in line:
                idx = line.index("=")
                line = line[: idx + 1]
                if extra_space:
                    line += " "
                if in_quotes:
                    v = '"{}"'.format(version)
                else:
                    v = "{}".format(version)
                fout.write(line + v + "\n")
            else:
                fout.write(line)


if __name__ == "__main__":
    major = 0
    minor = 4
    patch = 2

    mmp = f"{major}.{minor}.{patch}"
    mm = f"{major}.{minor}"

    kliff_dir = Path(__file__).parents[1]

    # update  __init__.py
    path = kliff_dir.joinpath("kliff", "__init__.py")
    update_version(mmp, path, "__version__", True, True)

    # update conf.py for docs
    path = kliff_dir.joinpath("docs", "source", "conf.py")
    update_version(mm, path, "version =", True, True)
    update_version(mmp, path, "release =", True, True)
