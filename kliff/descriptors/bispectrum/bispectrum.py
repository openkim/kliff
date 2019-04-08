import sys
import numpy as np
import kliff
from kliff.error import InputError
from kliff.neighbor import NeighborList
from kliff.dataset import Configuration
from kliff.descriptors.descriptor import Descriptor
from . import bs

logger = kliff.logger.get_logger(__name__)


class Bispectrum(Descriptor):
    """Bispectrum descriptor."""

    def __init__(self, jmax=3, *args, **kwargs):
        super(Bispectrum, self).__init__(*args, **kwargs)

        quadraticflag = 0

        rfac0 = 0.99363
        diagonalstyle = 3
        use_shared_arrays = 0
        rmin0 = 0
        switch_flag = 1
        bzero_flag = 0
        self._cdesc = bs.Bispectrum(rfac0, 2*jmax, diagonalstyle, use_shared_arrays,
                                    rmin0, switch_flag, bzero_flag)
