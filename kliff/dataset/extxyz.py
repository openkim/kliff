from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from kliff.error import InputError, KeyNotFoundError


def read_extxyz(
    filename: Path,
) -> Tuple[
    np.ndarray,
    List[str],
    np.ndarray,
    List[bool],
    Union[float, None],
    Union[np.ndarray, None],
    Union[List[float], None],
]:
    """
    Read atomic configuration stored in extended xyz file_format.

    Args:
        filename: filename to the extended xyz file

    Returns:
        cell: 3x3 array, supercell lattice vectors
        species: species of atoms
        coords: Nx3 array, coordinates of atoms
        PBC: periodic boundary conditions
        energy: potential energy of the configuration; `None` if not provided in file
        forces: Nx3 array, forces on atoms; `None` if not provided in file
        stress: 1D array of size 6, stress on the cell in Voigt notation; `None` if not
            provided in file
    """
    with open(filename, "r") as fin:
        lines = fin.readlines()

        try:
            natoms = int(lines[0].split()[0])
        except ValueError as e:
            raise InputError(f"{e}.\nCorrupted data at line 1 of file {filename}.")

        # lattice vector
        line = lines[1].replace("'", '"')
        cell = _parse_key_value(line, "Lattice", "float", 9, filename)
        cell = np.reshape(cell, (3, 3))

        # PBC
        PBC = _parse_key_value(line, "PBC", "int", 3, filename)

        # energy is optional
        try:
            in_quotes = _check_in_quotes(line, "Energy", filename)
            energy = _parse_key_value(line, "Energy", "float", 1, filename, in_quotes)[
                0
            ]
        except KeyNotFoundError:
            energy = None

        # stress is optional
        try:
            stress = _parse_key_value(line, "Stress", "float", 6, filename)
        except KeyNotFoundError:
            stress = None

        # body, species symbol, x, y, z (and fx, fy, fz if provided)
        species = []
        coords = []
        forces = []

        # if forces provided
        line = lines[2].strip().split()
        if len(line) == 4:
            has_forces = False
        elif len(line) == 7:
            has_forces = True
        else:
            raise InputError(f"Corrupted data at line 3 of file {filename}.")

        try:
            num_lines = 0
            for line in lines[2:]:
                num_lines += 1
                line = line.strip().split()
                if len(line) != 4 and len(line) != 7:
                    raise InputError(
                        f'Corrupted data at line {num_lines + 3} of file "{filename}".'
                    )
                if has_forces:
                    symbol, x, y, z, fx, fy, fz = line
                    species.append(symbol.lower().capitalize())
                    coords.append([float(x), float(y), float(z)])
                    forces.append([float(fx), float(fy), float(fz)])
                else:
                    symbol, x, y, z = line
                    species.append(symbol.lower().capitalize())
                    coords.append([float(x), float(y), float(z)])
        except ValueError as e:
            raise InputError(
                f"{e}.\nCorrupted data at line {num_lines + 3} of file {filename}."
            )

        if num_lines != natoms:
            raise InputError(
                f"Corrupted data file {filename}. Number of atoms is {natoms}, "
                f"whereas number of data lines is {num_lines}."
            )

        coords = np.asarray(coords)
        if has_forces:
            forces = np.asarray(forces)
        else:
            forces = None

        return cell, species, coords, PBC, energy, forces, stress


def write_extxyz(
    filename: Path,
    cell: np.ndarray,
    species: List[str],
    coords: np.ndarray,
    PBC: List[bool],
    energy: Optional[float] = None,
    forces: Optional[np.ndarray] = None,
    stress: Optional[List[float]] = None,
):
    """
    Write configuration info to a file in extended xyz file_format.

    Args:
        filename: filename to the extended xyz file
        cell: 3x3 array, supercell lattice vectors
        species: species of atoms
        coords: Nx3 array, coordinates of atoms
        PBC: periodic boundary conditions
        energy: potential energy of the configuration; If `None`, not write to file
        forces: Nx3 array, forces on atoms; If `None`, not write to file
        stress: 1D array of size 6, stress on the cell in Voigt notation; If `None`,
            not write to file
    """

    with open(filename, "w") as fout:

        # first line (number of atoms)
        natoms = len(species)
        fout.write("{}\n".format(natoms))

        # second line
        fout.write('Lattice="')
        for i, line in enumerate(cell):
            for j, item in enumerate(line):
                if i == 2 and j == 2:
                    fout.write('{:.15g}" '.format(item))
                else:
                    fout.write("{:.15g} ".format(item))

        PBC = [int(i) for i in PBC]
        fout.write('PBC="{} {} {}" '.format(PBC[0], PBC[1], PBC[2]))

        if energy is not None:
            fout.write('Energy="{:.15g}" '.format(energy))

        if stress is not None:
            fout.write('Stress="')
            for i, s in enumerate(stress):
                if i == 5:
                    fout.write('{:.15g}" '.format(s))
                else:
                    fout.write("{:.15g} ".format(s))

        properties = "Properties=species:S:1:pos:R:3"
        if forces is not None:
            properties += ":for:R:3\n"
        else:
            properties += "\n"
        fout.write(properties)

        # body
        for i in range(natoms):
            fout.write("{:2s} ".format(species[i]))
            fout.write(
                "{:23.15e} {:23.15e} {:23.15e} ".format(
                    coords[i][0], coords[i][1], coords[i][2]
                )
            )
            if forces is not None:
                fout.write(
                    "{:23.15e} {:23.15e} {:23.15e}".format(
                        forces[i][0], forces[i][1], forces[i][2]
                    )
                )
            fout.write("\n")


def _parse_key_value(
    line: str, key: str, dtype: str, size: int, filename: Path, in_quotes: bool = True
) -> List[Any]:
    """
    Given key, parse a string like ``other stuff key="value" other stuff`` to get value.

    If there is not space in value, the quotes `"` can be omitted.

    Args:
        line: The string line.
        key: Keyword to parse.
        dtype: Expected data type of value, `int` or `float`.
        size: Expected size of value.
        filename: File name where the line comes from.

    Returns:
        Values associated with key.
    """
    line = line.strip()
    key = _check_key(line, key, filename)
    try:
        value = line[line.index(key) :]
        if in_quotes:
            value = value[value.index('"') + 1 :]
            value = value[: value.index('"')]
        else:
            value = value[value.index("=") + 1 :]
            value = value.lstrip(" ")
            value += " "  # add an whitespace at end in case this is the last key
            value = value[: value.index(" ")]
        value = value.split()
    except Exception as e:
        raise InputError(f"{e}.\nCorrupted {key} data at line 2 of file {filename}.")

    if len(value) != size:
        raise InputError(
            f"Incorrect size of {key} at line 2 of file {filename};\n"
            f"required: {size}, provided: {len(value)}. Possibly, the quotes not match."
        )
    try:
        if dtype == "float":
            value = [float(i) for i in value]
        elif dtype == "int":
            value = [int(i) for i in value]
    except Exception as e:
        raise InputError(f"{e}.\nCorrupted {key} data at line 2 of file {filename}.")

    return value


def _check_key(line, key, filename):
    """
    Check whether a key or its lowercase counter part is in line.
    """
    if key not in line:
        key_lower = key.lower()
        if key_lower not in line:
            raise KeyNotFoundError(f"{key} not found at line 2 of file {filename}.")
        else:
            key = key_lower
    return key


def _check_in_quotes(line, key, filename):
    """
    Check whether ``key=value`` or ``key="value"`` in line.
    """
    key = _check_key(line, key, filename)
    value = line[line.index(key) :]
    value = value[value.index("=") + 1 :]
    value = value.lstrip(" ")
    if value[0] == '"':
        return True
    else:
        return False
