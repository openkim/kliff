import os


def update_version(version, path, key, in_quotes=False, extra_space=False):
    """Update version given in `key=version` structure."""
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
                    v = "'{}'".format(version)
                else:
                    v = "{}".format(version)
                fout.write(line + v + "\n")
            else:
                fout.write(line)


if __name__ == "__main__":
    kliff_v = "0.1.6"
    kliff_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(kliff_dir, "kliff", "__init__.py")
    update_version(kliff_v, path, "__version__", True, True)
