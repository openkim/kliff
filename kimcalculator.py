#!/usr/bin/env python

#============================================================================
# CDDL HEADER START
#
# The contents of this file are subject to the terms of the Common Development
# and Distribution License Version 1.0 (the "License").
#
# You can obtain a copy of the license at
# http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
# specific language governing permissions and limitations under the License.
#
# When distributing Covered Code, include this CDDL HEADER in each file and
# include the License file in a prominent location with the name LICENSE.CDDL.
# If applicable, add the following below this CDDL HEADER, with the fields
# enclosed by brackets "[]" replaced with your own identifying information:
#
# Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
#
# CDDL HEADER END
#
# Copyright (c) 2013, Regents of the University of Minnesota.
# All rights reserved.
#
# Contributors:
#    Matthew Bierbaum
#    Yanjiun Chen
#    Woosong Choi
#============================================================================

"""
OpenKIM Calculator for ASE

A calculator that uses models from the OpenKIM project
to find forces and energies of atom configurations.

We make use of the SWIG python KIM interface which is currently
available through github.com, soon to be included in the official
OpenKIM release.

Notes for work in progress:
    1. name pointers to KIM with "km_"
    2. need info on neighbor list
"""
import os
import glob
import numpy

from ase.calculators.interface import Calculator

import kimservice as ks
import kimneighborlist as kimnl

__version__ = '0.2'
__author__ = 'Matthew Bierbaum, Yanjiun Chen, Woosong Choi'


class KIMCalculator(Calculator):
    """
    KIMCalculator class which initializes an OpenKIM model
    and provides access to the registered compute method.

    Calculates energy, forces, stress, virial, hessian based
    on the capabilities of the model with which the calculator
    is initialized.
    """
    def __init__(self, modelname, kimfile='', search=True,
                 check_before_update=False, manual_update_only=False,
                 kimstring=""):
        """
        Creates a KIM calculator to ASE for a given modelname.

        Parameters
        ----------
        modelname: str
            The model with which the calculator is initialized

        kimfile: str
            If kimfile is present, it will use that file to initialize
            the KIM API. Takes precedence over search and kimstring.

        search: bool
            If search if True, it will look in the current
            folder for a .kim file, finally falling back to creating
            the test string from the configuration of atoms

        check_before_update: bool
            Indicates whether the calculator should first check that
            the Atoms class has changed in any way before requesting
            that the KIM model run `compute`.  Otherwise, when calling
            `get_`, always call `compute`.

            This saves time particularly for smaller systems:
                # ATOMS        CHECK     NOCHECK
                2    atoms:    65us      13.9ms
                54   atoms:    289us     13.8ms
                1458 atoms:    9.4ms     14.0ms

        manual_update_only: bool
            If True, the `compute` function (to calculate forces, energies) is
            only called manually.  In this case, use KIMCalculator.update(atoms)
            to update energies and forces.  If False, `compute` will be called
            based on check_before_update flag.

        kimstring: str
            A complete description of what our test requires as passed
            to the calculator as a proper .kim string.  This option
            overrides all other options.

        Returns
        -------
            out: KIMcalculator object

        """
        self.modelname = modelname
        self.check_before_update = check_before_update
        self.teststring = kimstring
        self.kimfile = kimfile
        self.manual_update_only = manual_update_only

        if not self.kimfile:
            if search:
                # look in the current directory for kim files
                potentials = glob.glob("./*.kim")
                for pot in potentials:
                    try:
                        with open(potentials[0]) as f:
                            self.teststring = f.read()
                    except Exception as e:
                        continue

        # initialize pointers for kim
        self.km_numberOfAtoms = None
        self.km_particleCharge = None
        self.km_energy = None
        self.km_forces = None
        self.km_particleEnergy = None
        self.km_virial = None
        self.km_particleVirial = None
        self.km_hessian = None

        # initialize ase atoms specifications
        self.pbc = None
        self.cell = None
        self.cell_orthogonal = None

        # the KIM object
        self.pkim = None
        self.uses_neighbors = None

    def set_atoms(self, atoms):
        """ Called by Atoms class in function set_calculator """
        if self.pkim:
            self.free_kim()
        self.init_kim(atoms)

    def init_kim(self, atoms):
        """ Initialize the KIM service for the current ASE atoms config """
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        self.cell_orthogonal = orthogonal(self.cell)

        if self.kimfile:
            # initialize with the KIM file in a standard directory
            status, self.pkim = ks.KIM_API_file_init(self.kimfile,
                    self.modelname)
        elif self.teststring:
            # initialize with the string we found in our kim file
            status, self.pkim = ks.KIM_API_init_str(self.teststring,
                    self.modelname)
        else:
            # if we haven't found a kim file yet, then go ahead and make a
            # KIM string which describes the capabilities that we require
            self.make_test_string(atoms)
            status, self.pkim = ks.KIM_API_init_str(self.teststring,
                    self.modelname)

        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)

        natoms = atoms.get_number_of_atoms()
        ntypes = len(set(atoms.get_atomic_numbers()))

        ks.KIM_API_allocate(self.pkim, natoms, ntypes)

        # set up the neighborlist as well, if necessary
        self.uses_neighbors = uses_neighbors(self.pkim)
        if self.uses_neighbors:
            kimnl.nbl_initialize(self.pkim)

        ks.KIM_API_model_init(self.pkim)

        # get pointers to model inputs
        self.km_numberOfAtoms = ks.KIM_API_get_data_ulonglong(self.pkim, "numberOfParticles")
        self.km_numberOfAtoms[0] = natoms
        self.km_numberAtomTypes = ks.KIM_API_get_data_int(self.pkim, "numberOfSpecies")
        self.km_numberAtomTypes[0] = ntypes
        self.km_atomTypes = ks.KIM_API_get_data_int(self.pkim, "particleSpecies")
        self.km_coordinates = ks.KIM_API_get_data_double(self.pkim, "coordinates")
        if checkIndex(self.pkim, "particleCharge") >= 0:
            self.km_particleCharge = ks.KIM_API_get_data_double(self.pkim, "particleCharge")
        if checkIndex(self.pkim, "particleSize") >= 0:
            self.km_particleSize = ks.KIM_API_get_data_double(self.pkim, "particleSize")

        # check what the model calculates and get model outputs
        if checkIndex(self.pkim, "energy") >= 0:
            self.km_energy = ks.KIM_API_get_data_double(self.pkim, "energy")
        if checkIndex(self.pkim, "forces") >= 0:
            self.km_forces = ks.KIM_API_get_data_double(self.pkim, "forces")
        if checkIndex(self.pkim, "particleEnergy") >= 0:
            self.km_particleEnergy = ks.KIM_API_get_data_double(self.pkim, "particleEnergy")
        if checkIndex(self.pkim, "virial") >= 0:
            self.km_virial = ks.KIM_API_get_data_double(self.pkim, "virial")
        if checkIndex(self.pkim, "particleVirial") >= 0:
            self.km_particleVirial = ks.KIM_API_get_data_double(self.pkim, "particleVirial")
        if checkIndex(self.pkim, "hessian") >= 0:
            self.km_hessian = ks.KIM_API_get_data_double(self.pkim, "hessian")

    def free_kim(self):
        if self.uses_neighbors:
            kimnl.nbl_cleanup(self.pkim)
        ks.KIM_API_model_destroy(self.pkim)
        ks.KIM_API_free(self.pkim)

        self.pkim = None

    def make_test_string(self, atoms, tmp_name="test_name"):
        """ Makes string if it doesn't exist, if exists just keeps it as is """
        if not self.teststring or self.cell_BC_changed(atoms):
            self.teststring = make_kimscript(tmp_name, self.modelname, atoms)

    def cell_BC_changed(self, atoms):
        """
        Check whether BC has changed and cell orthogonality has changed
        because we might want to change neighbor list generator method
        """
        return ((self.pbc != atoms.get_pbc()).any() or
                 self.cell_orthogonal != orthogonal(atoms.get_cell()))

    def calculation_required(self, atoms, quantities):
        """
        Check whether or not the atoms configuration has
        changed and we need to recalculate..
        """
        return (self.km_energy is None or
               (self.km_numberOfAtoms[0] != atoms.get_number_of_atoms()) or
               (self.km_atomTypes[:] != atoms.get_atomic_numbers()).any() or
               (self.km_coordinates[:] != atoms.get_positions().flatten()).any() or
               (self.pbc != atoms.get_pbc()).any() or
               (self.cell != atoms.get_cell()).any())

    def update(self, atoms):
        """
        Connect the KIM pointers to values in the ase atoms class
        set up neighborlist and perform calculation
        """
        # here we only reinitialize the model if the number of Atoms /
        # types of atoms have changed, or if the model is uninitialized
        natoms = atoms.get_number_of_atoms()
        ntypes = len(set(atoms.get_atomic_numbers()))

        if (self.km_numberOfAtoms[0] != natoms or
            self.km_numberAtomTypes[0] != ntypes or
            self.cell_BC_changed(atoms)):
            self.set_atoms(atoms)

        if (not self.check_before_update or
            (self.check_before_update and self.calculation_required(atoms, ""))):
            # if the calculation is required we proceed to set the values
            # of the standard things each model and atom class has
            self.km_numberOfAtoms[0] = natoms
            self.km_numberAtomTypes[0] = ntypes
            self.km_coordinates[:] = atoms.get_positions().flatten()
            if self.km_particleCharge:
                km_particleCharge[:] = atoms.get_charges()

            # fill the proper chemical identifiers
            symbols = atoms.get_chemical_symbols()
            for i in range(natoms):
                self.km_atomTypes[i] = ks.KIM_API_get_species_code(self.pkim, symbols[i])

            # build the neighborlist (type depends on model set by pkim)
            if self.uses_neighbors:
                kimnl.nbl_set_cell(atoms.get_cell().flatten(), atoms.get_pbc().flatten().astype('int8'))
                kimnl.nbl_build_neighborlist(self.pkim)
            ks.KIM_API_model_compute(self.pkim)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_energy is not None:
            return self.km_energy.copy()[0]
        else:
            raise SupportError("energy")

    def get_potential_energies(self, atoms):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_particleEnergy is not None:
            particleEnergies = self.km_particleEnergy
            return particleEnergies.copy()
        else:
            raise SupportError("potential energies")

    def get_forces(self, atoms):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_forces is not None:
            forces = self.km_forces.reshape((self.km_numberOfAtoms[0], 3))
            return forces.copy()
        else:
            raise SupportError("forces")

    def get_stress(self, atoms):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_virial is not None:
            return self.km_virial.copy()
        else:
            raise SupportError("stress")

    def get_stresses(self, atoms):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_particleVirial is not None:
            return self.km_particleVirial.copy()
        else:
            raise SupportError("stress per particle")

    def get_hessian(self, atoms):
        if not self.manual_update_only:
            self.update(atoms)
        if self.km_hessian is not None:
            return self.km_hessian.copy()
        else:
            raise SupportError("hessian")

    def get_NBC_method(self):
        if self.pkim:
            return ks.KIM_API_get_NBC_method(self.pkim)

    def set_ghosts(self, ghosts):
        if self.uses_neighbors:
            kimnl.nbl_set_ghosts(ghosts, self.get_NBC_method() == "NEIGH_PURE_H")
            self.uses_ghosts = True

    def __del__(self):
        """ Garbage collects the KIM API objects automatically """
        if self.pkim:
            if self.uses_neighbors:
                kimnl.nbl_cleanup(self.pkim)
            ks.KIM_API_free(self.pkim)
        self.pkim = None

    def __str__(self):
        return "KIMCalculator(" + self.modelname + ")"


def make_kimscript(testname, modelname, atoms):
    """
    Creates a valid KIM file according to the needs imposed by the atoms
    object including deciding which neighborlist varieties are passable

    Parameters
    ----------
    testname: str
        the name of the test which this script is creating

    atoms: Atom object
        the ASE Atoms object which determines the test requirements

    Returns
    -------
    kimstring: str
        a string version of the KIM file for this atoms object
    """
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    cell_orthogonal = orthogonal(cell)

    kimstr = "TEST_NAME := " + testname + "\n"

    # BASE UNIT LINES
    unit_length = "A"
    unit_energy = "eV"
    unit_charge = "e"
    unit_temperature = "K"
    unit_time = "ps"

    kimstr += "KIM_API_Version := 1.6.0\n"
    kimstr += "Unit_length := " + unit_length + "\n"
    kimstr += "Unit_energy := " + unit_energy + "\n"
    kimstr += "Unit_charge := " + unit_charge + "\n"
    kimstr += "Unit_temperature := " + unit_temperature + "\n"
    kimstr += "Unit_time := " + unit_time + "\n"

    # SUPPORTED_ATOM/PARTICLE_TYPES
    kimstr += "PARTICLE_SPECIES: \n"

    # check ASE atoms class for which atoms it has
    acodes = set(atoms.get_atomic_numbers())
    asymbols = set(atoms.get_chemical_symbols())

    for code, symbol in zip(list(acodes), list(asymbols)):
        kimstr += symbol + " spec " + str(code) + "\n"

    # CONVENTIONS
    kimstr += "CONVENTIONS:\n"

    # note: by default the convention for python is Zero-based lists
    kimstr += "ZeroBasedLists  flag\n"
    kimstr += "Neigh_IterAccess  flag\n"
    kimstr += "Neigh_LocaAccess  flag\n"
    kimstr += "Neigh_BothAccess  flag\n"

    # Neighbor list and Boundary Condition (NBC) methods
    if pbc.any():
        kimstr += "NEIGH_RVEC_F flag \n"
        kimstr += "NEIGH_RVEC_H flag \n"

        # we can have OPBC if the cell is not slanty
        if cell_orthogonal:
            kimstr += "MI_OPBC_F flag \n"
            kimstr += "MI_OPBC_H flag \n"
    else:
        kimstr += "NEIGH_RVEC_H flag\n"
        kimstr += "NEIGH_RVEC_F flag\n"
        kimstr += "NEIGH_PURE_H flag\n"
        kimstr += "NEIGH_PURE_F flag\n"
        kimstr += "MI_OPBC_F flag \n"
        kimstr += "MI_OPBC_H flag \n"
        kimstr += "CLUSTER flag \n"

    # MODEL_INPUT section
    kimstr += "MODEL_INPUT:\n"
    kimstr += "numberOfParticles  integer  none  []\n"
    kimstr += "numberOfSpecies integer  none  []\n"
    kimstr += "particleSpecies  integer  none  [numberOfParticles]\n"
    kimstr += "coordinates  double  length  [numberOfParticles,3]\n"

    if atoms.get_charges().any():
        kimstr += "particleCharge  double  charge  [numberOfParticles]\n"
    kimstr += "numberContributingParticles  integer  none  []\n"
    kimstr += "boxSideLengths  double  length  [3]\n"

    kimstr += "get_neigh  method  none []\n"
    kimstr += "neighObject  pointer  none  []\n"

    # MODEL_OUTPUT section
    # Here, we choose to match the model to allow for easy matching, but
    # we can always raise support errors later when it turns out the model
    # we used can perform a certain task
    status, km_pmdl = ks.KIM_API_model_info(modelname)

    kimstr += "MODEL_OUTPUT: \n"
    if checkIndex(km_pmdl, "compute") >= 0:
        kimstr += "compute  method  none  []\n"
    if checkIndex(km_pmdl, "reinit") >= 0:
        kimstr += "reinit  method  none  []\n"
    if checkIndex(km_pmdl, "destroy") >= 0:
        kimstr += "destroy  method  none  []\n"
    if checkIndex(km_pmdl, "cutoff") >= 0:
        kimstr += "cutoff  double  length  []\n"
    if checkIndex(km_pmdl, "energy") >= 0:
        kimstr += "energy  double  energy  []\n"
    if checkIndex(km_pmdl, "forces") >= 0:
        kimstr += "forces  double  force  [numberOfParticles,3]\n"
    if checkIndex(km_pmdl, "particleEnergy") >= 0:
        kimstr += "particleEnergy  double  energy  [numberOfParticles]\n"
    if (checkIndex(km_pmdl, "virial") >= 0 or checkIndex(km_pmdl, "process_dEdr") >=0):
        kimstr += "virial  double  energy  [6]\n"
    if (checkIndex(km_pmdl, "particleVirial") >= 0 or checkIndex(km_pmdl, "process_dEdr") >=0):
        kimstr += "particleVirial  double  energy  [numberOfParticles,6]\n"
    if (checkIndex(km_pmdl, "hessian") >= 0 or
	   (checkIndex(km_pmdl, "process_dEdr") >= 0 and checkIndex(km_pmdl, "process_d2Edr2") >= 0)):
        kimstr += "hessian  double  pressure  [numberOfParticles,numberOfParticles,3,3]\n"
    return kimstr


def orthogonal(cell):
    return ((abs(numpy.dot(cell[0], cell[1])) +
             abs(numpy.dot(cell[0], cell[2])) +
             abs(numpy.dot(cell[1], cell[2]))) < 1e-8)


def uses_neighbors(pkim):
    # to get model units, inputs, outputs, options we call KIM_API_model_info
    if ks.KIM_API_get_NBC_method(pkim) == "CLUSTER":
        return 0
    return 1


def checkIndex(pkim, variablename):
    try:
        index = ks.KIM_API_get_index(pkim, variablename)
    except:
        index = -1
    return index


def listmodels():
    try:
        kimdir = os.environ['KIM_MODELS_DIR']
    except:
        try:
            kimdir = os.path.join(os.environ['KIM_DIR'], "MODELS")
        except:
            print "No KIM_MODELS_DIR set"
            return

    models = []
    for model in glob.glob(os.path.join(kimdir, '*')):
        if os.path.isdir(model):
            models.append(os.path.basename(model))
    return models


class SupportError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + " computation not supported by model"


class InitializationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value) + " initialization failed"
