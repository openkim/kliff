import kimservice as ks
import kimneighborlist as kimnl

class KIMobject:
    ''' 
    KIMobject class which initializes an OpenKIM object that stores the Model 
    information and provides access to the registered compute method.
    ''' 
    def __init__(self, modelname, conf):
        '''
        Creates a KIM calculator to ASE for a given modelname.

        Parameters
        ----------
        modelname: str
            the KIM Model upon which the KIM object is built 

        conf: Config object in which the atoms information are stored

        kimstring: str
            descriptor string (functions the same as descriptor.kim file) of the
            this test, which is used to match the KIM Model

        Returns
        -------
            out: KIM object
        '''
        
        # class members
        self.modelname = modelname
        self.conf = conf

        # initialize pointers for kim
        self.km_nparticles = None
        self.km_nspecies = None 
        self.km_particleSpecies = None
        self.km_coords = None

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

#    def set_atoms(self, atoms):
#        """ Called by Atoms class in function set_calculator """
#        if self.pkim:
#            self.free_kim()
#        self.initialize(atoms)
#

    def initialize(self):
        ''' Initialize the KIM object for self.conf''' 

#        self.pbc = atoms.get_pbc()
#        self.cell = atoms.get_cell()
#        self.cell_orthogonal = orthogonal(self.cell)
#
#        if self.kimfile:
#            # initialize with the KIM file in a standard directory
#            status, self.pkim = ks.KIM_API_file_init(self.kimfile,
#                    self.modelname)
#        elif self.teststring:
#            # initialize with the string we found in our kim file
#            status, self.pkim = ks.KIM_API_init_str(self.teststring,
#                    self.modelname)
#        else:
#            # if we haven't found a kim file yet, then go ahead and make a
#            # KIM string which describes the capabilities that we require
#            self.make_test_string(atoms)
#            status, self.pkim = ks.KIM_API_init_str(self.teststring,
       
        # inquire information from the conf
        particleSpecies = self.conf.get_species()
        species = set(particleSpecies)
        nspecies = len(species)
        nparticles = self.conf.get_num_atoms()
        coords = self.conf.get_coords()

        kim_str = generate_kimstr(self.modelname, species) 
        status, self.pkim = ks.KIM_API_init_str(kimstr, self.modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)

#NOTE the following two status may not work, Matt's code does not use it
# yes it does not work
        status = ks.KIM_API_allocate(self.pkim, nparticles, nspecies)
#        if ks.KIM_STATUS_OK != status:
#            ks.KIM_API_report_error('KIM_API_allocate', status)
#            raise InitializationError(self.modelname)
#
        status = ks.KIM_API_model_init(self.pkim)
#        if ks.KIM_STATUS_OK != status:
#            ks.KIM_API_report_error('KIM_API_model_init', status)
#            raise InitializationError(self.modelname)
#
        # get pointers to model inputs
        self.km_nparticles = ks.KIM_API_get_data_ulonglong(self.pkim, "numberOfParticles")
        self.km_nparticles[0] = nparticles
        self.km_nspecies = ks.KIM_API_get_data_int(self.pkim, "numberOfSpecies")
        self.km_nspecies[0] = nspecies
        self.km_particleSpecies = ks.KIM_API_get_data_int(self.pkim, "particleSpecies")
        self.km_coords = ks.KIM_API_get_data_double(self.pkim, "coordinates")

#NOTE we may need numberOfcontributingAtoms to use half list 

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

        # copy particle species
        for i,s in enumerate(particleSpecies):
            self.km_particleSpecies[i] = ks.KIM_API_get_species_code(self.pkim, s)

        # copy coordinates 
        for i,c in enumerate(coords):
            self.km_coords[i] = c 

#NOTE
# if we want to use MIOPBC, we need to add something below

        # set up the neighborlist 
        kimnl.nbl_initialize(self.pkim)





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








def generate_kimstr(modelname, species):
    '''
    Creates a valid KIM file that will be used to initialize the KIM object.

    Parameters
    ----------
    modelname: KIM Model name

    Tset: TrainingSet object
        the training set object from which we get the atom species 

    Returns
    -------
    kimstring: str
        a string of the KIM file for the configuration object
    '''
    
    # version and units
    kimstr  = 'KIM_API_Version := 1.7.0\n'
    kimstr += 'Unit_length := A\n'
    kimstr += 'Unit_energy := eV\n'
    kimstr += 'Unit_charge := e\n'
    kimstr += 'Unit_temperature := K\n'
    kimstr += 'Unit_time := ps\n'

    # particle species
    # 'code' does not matter, so just give it 0 
    kimstr += 'PARTICLE_SPECIES:\n'
    kimstr += '# Symbol/name    Type    code\n'
    for s in species:
        kimstr += s+'  spec    0\n' 

    # conversions 
    kimstr += 'CONVENTIONS:\n'
    kimstr += 'ZeroBasedLists   flag\n'
    kimstr += 'Neigh_LocaAccess flag\n'
    kimstr += 'Neigh_IterAccess flag\n'
    kimstr += 'Neigh_BothAccess flag\n'
    kimstr += 'NEIGH_RVEC_H flag\n'
    kimstr += 'NEIGH_RVEC_F flag\n'
    kimstr += 'NEIGH_PURE_H flag\n'
    kimstr += 'NEIGH_PURE_F flag\n'
    kimstr += 'MI_OPBC_F    flag\n'
    kimstr += 'MI_OPBC_H    flag\n'
    kimstr += 'CLUSTER      flag\n'

    # model input
    kimstr += 'MODEL_INPUT:\n'
    kimstr += 'numberOfParticles  integer  none    []\n'
    kimstr += 'numberOfSpecies    integer  none    []\n'
    kimstr += 'particleSpecies    integer  none    [numberOfParticles]\n'
    kimstr += 'coordinates        double   length  [numberOfParticles,3]\n'
    kimstr += 'boxSideLengths     double   length  [3]\n'
    kimstr += 'numberContributingParticles integer none  []\n'
    kimstr += 'get_neigh          method   none    []\n'
    kimstr += 'neighObject        pointer  none    []\n'

    # model output
    # create a temporary object to inquire the info
    status, kimmdl = ks.KIM_API_model_info(modelname) 
    kimstr += "MODEL_OUTPUT:\n"
    if checkIndex(kimmdl, 'compute') >= 0:
        kimstr += 'compute  method  none  []\n'
    if checkIndex(kimmdl, 'reinit') >= 0:
        kimstr += 'reinit   method  none  []\n'
    if checkIndex(kimmdl, 'destroy') >= 0:
        kimstr += 'destroy  method  none  []\n'
    if checkIndex(kimmdl, 'cutoff') >= 0:
        kimstr += 'cutoff  double  length  []\n'
    if checkIndex(kimmdl, 'energy') >= 0:
        kimstr += 'energy  double  energy  []\n'
    if checkIndex(kimmdl, 'forces') >= 0:
        kimstr += 'forces  double  force  [numberOfParticles,3]\n'
    if checkIndex(kimmdl, 'particleEnergy') >= 0:
        kimstr += 'particleEnergy  double  energy  [numberOfParticles]\n'
    if (checkIndex(kimmdl, 'virial') >= 0 or checkIndex(kimmdl, 'process_dEdr') >=0):
        kimstr += 'virial  double  energy  [6]\n'
    if (checkIndex(kimmdl, 'particleVirial') >= 0 or checkIndex(kimmdl, 'process_dEdr') >=0):
        kimstr += 'particleVirial  double  energy  [numberOfParticles,6]\n'
    if (checkIndex(kimmdl, 'hessian') >= 0 or
	   (checkIndex(kimmdl, 'process_dEdr') >= 0 and checkIndex(kimmdl, 'process_d2Edr2') >= 0)):
        kimstr += 'hessian  double  pressure  [numberOfParticles,numberOfParticles,3,3]\n'

#NOTE if we free it, segfault error will occur; Ask Matt whether we don't need to free?
#NOTE in kimcalculator, it is not freed
    # free KIM object
    #ks.KIM_API_model_destroy(kimmdl)
    #ks.KIM_API_free(kimmdl)

    return kimstr


def checkIndex(pkim, variablename):
    '''
    Check whether a variable exists in the KIM object.
    '''
    try:
        index = ks.KIM_API_get_index(pkim, variablename)
    except:
        index = -1
    return index


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



if __name__ == '__main__':
   
    # test generate_kimstr()
    from training import TrainingSet
    tset = TrainingSet()
    tset.read('./training_set')
    #modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    configs = tset.get_configs()
    
    # test generate_kimstr
    species = set(configs[0].get_species())
    kimstr = generate_kimstr(modelname, species)
    print kimstr

    # initialize objects
    KIMobj = KIMobject(modelname, configs[0]) 
    KIMobj.initialize()
    
    
    
