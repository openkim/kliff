import kimservice as ks
import kimneighborlist as kimnl
import numpy as np

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

#        # initialize ase atoms specifications
#        self.pbc = None
#        self.cell = None
#        self.cell_orthogonal = None

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
        cell = self.conf.get_cell().flatten()

        kimstr = generate_kimstr(self.modelname, cell, species) 
        status, self.pkim = ks.KIM_API_init_str(kimstr, self.modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)

        ks.KIM_API_allocate(self.pkim, nparticles, nspecies)

        ks.KIM_API_model_init(self.pkim)

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
# if we want to use MIOPBC, we need to add something below  see potfit


# NOTE see universal test about how to set up neighborlist
# we still need to still ghost if we want to use neigh_pure
# or possibly, we can use periodic boundary conditions for neigh_pure

        # set up neighbor list 
        self.set_neighborlist()


    def compute(self):
        ks.KIM_API_model_compute(self.pkim)


    def set_neighborlist(self):
        '''
        Initialize neighborlist once and for all.
        '''
        PBC = self.conf.get_pbc()
        cell = self.conf.get_cell().flatten()
        NBC = self.get_NBC_method()
        if NBC == 'CLUSTER':
            self.uses_neighbors = False
        else:
            self.uses_neighbors = True
        if self.uses_neighbors == True:
            kimnl.nbl_initialize(self.pkim)
            kimnl.nbl_set_cell(cell, PBC)
            kimnl.nbl_build_neighborlist(self.pkim)




#NOTE
#But we may want to move KIM_API_model_destroy to __del__

# this may not be needed, since the the __del__ will do the free automatically
    def free_kim(self):
        if self.uses_neighbors:
            kimnl.nbl_cleanup(self.pkim)
        ks.KIM_API_model_destroy(self.pkim)
        ks.KIM_API_free(self.pkim)
        self.pkim = None

# NOTE
# we may want to define a function `update` to publish parameters  

    def get_coords(self):
        if self.km_coords is not None:
            return self.km_coords.copy()
        else:
            raise SupportError("forces")


    def get_potential_energy(self):
        if self.km_energy is not None:
            return self.km_energy.copy()[0]
        else:
            raise SupportError("energy")

    def get_particle_energy(self):
        if self.km_particleEnergy is not None:
            return self.km_particleEnergy.copy()
        else:
            raise SupportError("partile energy")

    def get_forces(self):
        if self.km_forces is not None:
            return self.km_forces.copy()
        else:
            raise SupportError("forces")

#    def get_stress(self, atoms):
#        if self.km_virial is not None:
#            return self.km_virial.copy()
#        else:
#            raise SupportError("stress")
#
#    def get_stresses(self, atoms):
#        if self.km_particleVirial is not None:
#            return self.km_particleVirial.copy()
#        else:
#            raise SupportError("stress per particle")
#
#    def get_hessian(self, atoms):
#        if self.km_hessian is not None:
#            return self.km_hessian.copy()
#        else:
#            raise SupportError("hessian")
#
    def get_NBC_method(self):
        if self.pkim:
            return ks.KIM_API_get_NBC_method(self.pkim)


    def __del__(self):
        """ Garbage collects the KIM API objects automatically """
        if self.pkim:
            if self.uses_neighbors:
                kimnl.nbl_cleanup(self.pkim)
            ks.KIM_API_free(self.pkim)
        self.pkim = None




def generate_kimstr(modelname, cell, species):
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
    if orthogonal(cell):
        kimstr += 'MI_OPBC_H    flag\n'
        kimstr += 'MI_OPBC_F    flag\n'
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


def orthogonal(cell):
    '''
    Check whether the supercell is orthogonal. 
    '''
    return ((abs(np.dot(cell[0], cell[1])) +
             abs(np.dot(cell[0], cell[2])) +
             abs(np.dot(cell[1], cell[2]))) < 1e-8)


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
    #modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
    
    #tset.read('./training_set')
    #tset.read('./training_set/T150_training_1000.xyz')
    tset.read('./develop_test/config.txt_20x20')
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    
    #tset.read('./develop_test/training_set_Si.xyz')
    #modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
    
    configs = tset.get_configs()
    

    # initialize objects
    KIMobj = KIMobject(modelname, configs[0]) 
    KIMobj.initialize()
    KIMobj.compute()  
    
    print KIMobj.get_NBC_method()
    print KIMobj.get_potential_energy() 
    coords = KIMobj.get_coords()
    forces = KIMobj.get_forces() 
    coords = np.reshape(coords, (len(coords)/3, 3))
    forces = np.reshape(forces, (len(forces)/3, 3))
#    for r,f in zip(coords,forces):
#        for i in r:
#            print '{:15.7e}'.format(i),
#        for j in f:
#            print '{:15.7e}'.format(j),
#        print

#    print KIMobj.km_coords 

    ks.KIM_API_print(KIMobj.pkim)
