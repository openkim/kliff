import numpy as np
from kliff.error import InputError, KeyNotFoundError


def read_extxyz(fname):
    """Read atomic configuration stored in extended xyz format.

    Parameters
    ----------
    fname: str
        name of the extended xyz file

    Returns
    -------
    cell: 2D darray of shape(3,3)
        supercell lattice vectors

    PBC: list of 3 bool
        periodic boundary condictions

    species: list of N str, where N is the number of atoms
        species of atoms

    coords: 2D array of shape (N, 3)
        coordinates of atoms

    energy: float (or None if not provided in file)
        potential energy of the configuration

    forces: 2D array of shape (N, 3) (or None if not provided in file)
        forces on atoms

    stress: list of 6 float (or None if not provided in file)
        stress on the cell in Voigt notation
    """
    with open(fname, 'r') as fin:
        lines = fin.readlines()

        try:
            natoms = int(lines[0].split()[0])
        except ValueError as e:
            raise InputError(
                '{}.\nCorrupted data at line 1 of file "{}".'.format(e, fname)
            )

        # lattice vector, PBC, energy, and stress
        line = lines[1].replace("'", '"')
        cell = parse_key_value(line, 'Lattice', 'float', 9, fname)
        cell = np.reshape(cell, (3, 3))
        PBC = parse_key_value(line, 'PBC', 'int', 3, fname)
        # energy is optional
        try:
            in_quotes = check_in_quotes(line, 'Energy', fname)
            energy = parse_key_value(line, 'Energy', 'float', 1, fname, in_quotes)[0]
        except KeyNotFoundError:
            energy = None
        # stress is optional
        try:
            stress = parse_key_value(line, 'Stress', 'float', 6, fname)
        except KeyNotFoundError:
            stress = None

        # body, species symbol, x, y, z (and fx, fy, fz if provided)
        species = []
        coords = []
        forces = []
        # is forces provided
        line = lines[2].strip().split()
        if len(line) == 4:
            has_forces = False
        elif len(line) == 7:
            has_forces = True
        else:
            raise InputError('Corrupted data at line 3 of file "{}" .'.format(fname))

        try:
            num_lines = 0
            for line in lines[2:]:
                num_lines += 1
                line = line.strip().split()
                if len(line) != 4 and len(line) != 7:
                    raise InputError(
                        'Corrupted data at line {} of file "{}".'.format(
                            num_lines + 3, fname
                        )
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
                '{}.\nCorrupted data at line {} of file "{}".'.format(
                    e, num_lines + 3, fname
                )
            )

        if num_lines != natoms:
            raise InputError(
                'Corrupted data file "{}". Number of atoms is "{}", '
                'whereas number of data lines is "{}".'.format(fname, natoms, num_lines)
            )

        species = np.asarray(species)
        coords = np.asarray(coords)
        if has_forces:
            forces = np.asarray(forces)
        else:
            forces = None
        return cell, PBC, species, coords, energy, forces, stress


def write_extxyz(
    fname, cell, PBC, species, coords, energy=None, forces=None, stress=None
):
    """
    Write configuration info to a file in extended xyz format.

    Parameters
    ----------
    fname: str
        name of the written file

    cell: 2D darray of shape(3,3)
        supercell lattice vectors

    PBC: list of 3 bool
        periodic boundary condictions

    species: list of N str, where N is the number of atoms
        species of atoms

    coords: 2D array of shape (N, 3)
        coordinates of atoms

    energy: float (optional)
        potential energy of the configuration

    forces: 2D array of shape (N, 3) (optional)
        forces on atoms

    stress: list of 6 float (optional)
        stress on the cell in Voigt notation
    """

    with open(fname, 'w') as fout:

        # first line (number of atoms)
        natoms = len(species)
        fout.write('{}\n'.format(natoms))

        # second line
        fout.write('Lattice="')
        for i, line in enumerate(cell):
            for j, item in enumerate(line):
                if i == 2 and j == 2:
                    fout.write('{:.15g}" '.format(item))
                else:
                    fout.write('{:.15g} '.format(item))

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
                    fout.write('{:.15g} '.format(s))

        properties = 'Properties=species:S:1:pos:R:3'
        if forces is not None:
            properties += ':for:R:3\n'
        else:
            properties += '\n'
        fout.write(properties)

        # body
        for i in range(natoms):
            fout.write('{:2s} '.format(species[i]))
            fout.write(
                '{:23.15e} {:23.15e} {:23.15e} '.format(
                    coords[i][0], coords[i][1], coords[i][2]
                )
            )
            if forces is not None:
                fout.write(
                    '{:23.15e} {:23.15e} {:23.15e}'.format(
                        forces[i][0], forces[i][1], forces[i][2]
                    )
                )
            fout.write('\n')


def check_key(line, key, fname):
    """Check whether a key or its lowercase counter part is in line."""
    if key not in line:
        key_lower = key.lower()
        if key_lower not in line:
            raise KeyNotFoundError(
                '"{}" not found at line 2 of file "{}".'.format(key, fname)
            )
        else:
            key = key_lower
    return key


def check_in_quotes(line, key, fname):
    """Check wheter ``key=value`` or ``key="value"`` in line."""
    key = check_key(line, key, fname)
    value = line[line.index(key) :]
    value = value[value.index('=') + 1 :]
    value = value.lstrip(' ')
    if value[0] == '"':
        return True
    else:
        return False


def parse_key_value(line, key, dtype, size, fname, in_quotes=True):
    """Given key, parse a string like ``other stuff key="value" other stuff``
    to get value.

    If there is not space in value, the quotes `"` can be omitted.

    Parameters
    ----------
    line: str
        The string line.

    key: str
        Keyword to parse.

    dtype: str
        Expected data type of value, `int` or `float`.

    size: int
        Expected size of value.

    fname: str
        File name where the line comes from.

    Return
    ------
    list
        Values associated with key.
    """
    line = line.strip()
    key = check_key(line, key, fname)
    try:
        value = line[line.index(key) :]
        if in_quotes:
            value = value[value.index('"') + 1 :]
            value = value[: value.index('"')]
        else:
            value = value[value.index('=') + 1 :]
            value = value.lstrip(' ')
            value += ' '  # add an whitespace at end in case this is the last key
            value = value[: value.index(' ')]
        value = value.split()
    except Exception as e:
        raise InputError(
            '{}.\nCorrupted "{}" data at line 2 of file "{}".'.format(e, key, fname)
        )

    if len(value) != size:
        raise InputError(
            'Incorrect size of "{}" at line 2 of file "{}";\n'
            'required: {}, provided: {}. Possibly, the quotes do not '
            'match.'.format(key, fname, size, len(value))
        )
    try:
        if dtype == 'float':
            value = [float(i) for i in value]
        elif dtype == 'int':
            value = [int(i) for i in value]
    except Exception as e:
        raise InputError(
            '{}.\nCorrupted "{}" data at line 2 of file "{}".'.format(e, key, fname)
        )

    return np.asarray(value)
