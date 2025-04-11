def get_bs_size(twojmax: int, diagonal: int) -> int:
    """
    Return the size of bispectrum descriptor.

    Args:
        twojmax: 2x  Maximum angular momentum for bispectrum.
        diagonal: 0, 1, 2, or 3. 0: full, 1: lower triangle, 2: diagonal, 3: upper triangle.
    """
    N = 0
    for j1 in range(0, twojmax + 1):
        if diagonal == 2:
            N += 1
        elif diagonal == 1:
            for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                N += 1
        elif diagonal == 0:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    N += 1
        elif diagonal == 3:
            for j2 in range(0, j1 + 1):
                for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        N += 1
    return N


def get_soap_size(nmax: int, lmax: int, n_species: int) -> int:
    """
    Return the size of SOAP descriptor.

    Args:
        nmax: Maximum radial quantum number.
        lmax: Maximum angular momentum.
        n_species: Number of species.
    """
    return int(((n_species + 1) * n_species) / 2 * (nmax * (nmax + 1)) * (lmax + 1) / 2)
