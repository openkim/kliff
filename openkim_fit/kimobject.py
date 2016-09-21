import numpy as np
from collections import OrderedDict
import kimservice as ks 
import kimneighborlist as kimnl
from utils import generate_kimstr
from utils import checkIndex 

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

        # optimization parameters
        self.params = OrderedDict()
        self.params_index = [] 

#    def set_atoms(self, atoms):
#        """ Called by Atoms class in function set_calculator """
#        if self.pkim:
#            self.free_kim()
#        self.initialize(atoms)
#

    def initialize(self):
        ''' Initialize the KIM object for the configuration of atoms in self.conf.''' 

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
        cell = self.conf.get_cell()

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
#        if checkIndex(self.pkim, "virial") >= 0:
#            self.km_virial = ks.KIM_API_get_data_double(self.pkim, "virial")
#        if checkIndex(self.pkim, "particleVirial") >= 0:
#            self.km_particleVirial = ks.KIM_API_get_data_double(self.pkim, "particleVirial")
#        if checkIndex(self.pkim, "hessian") >= 0:
#            self.km_hessian = ks.KIM_API_get_data_double(self.pkim, "hessian")
#
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

    def map_opt_index(self, initial_params):
        '''
        Set up the map between initial parameters that are read in from input
        and the parameters that are to be optimzied. The latter is a one-dimensional
        list that is going to be used by the optimized directly. "index" means
        the slot of of a parameter in the one-dimensional list. For example, 
        given the following potential model paramters in the input file:

        PARAM_FREE_A
        1.1  fix 
        KIM  2.2  3.3
        
        PARAM_FREE_B
        4.4
        KIM  fix
        5.5  2.2  6.6 
       
        and a one-dimensional list opt_x, KIM in PARAM_FREE_A will have index 0 in
        opt_x, and 4.4 and 5.5 in PARAM_FREE_B will have indices 1 and 2, respectively.
        '''
        initial_param_names = initial_params.get_names()
        for name in initial_param_names:
            value = ks.KIM_API_get_data_double(self.pkim, name)
            initial_value = initial_params.get_value(name) 
            size          = initial_params.get_size(name) 
            lower_bound   = initial_params.get_lower_bound(name) 
            upper_bound   = initial_params.get_upper_bound(name) 
            fix           = initial_params.get_fix(name) 
            
            # copy intial guesses to KIM
            # should not do: value = initial_value.copy() since this will simply 
            # reassign value to another object. Here, we should regard value as 
            # a C-type pointer registered in the KIM object
            for i in range(size):
                value[i] = initial_value[i]

            self.params[name] = {'size':size, 'value':value,
                                 'lower_bound':lower_bound,
                                 'upper_bound':upper_bound,
                                 'fix':fix}

            # create index 
            for i in range(size):
                if not fix[i]:
                    tmp_idx = {'name':name, 'value_slot':i}
                    self.params_index.append(tmp_idx)


            # create index 
#            for i in range(size):
#                if not fix[i]:
#                    index[i] = idx
#                    idx += 1 
#                else:
#                    index[i] = None
#
#            self.params[name] = {'size':size, 'value':value,
#                                 'lower_bound':lower_bound,
#                                 'upper_bound':upper_bound,
#                                 'fix':fix, 'index':index}
#

    def publish_params(self, opt_x):
        '''
        Update parameters from optimzier to KIM object and then re-initiate the model.
        '''
        for i,val in enumerate(opt_x):
            name = self.params_index[i]['name']
            value_slot = self.params_index[i]['value_slot']
            self.params[name]['value'][value_slot] = val
        ks.KIM_API_model_reinit(self.pkim)

    def get_opt_x0(self):
        '''
        Nest all parameter values (except the fix ones) to a list. And this will
        be fed to the optimizer as the starting parameters.
        '''
        opt_x0 = [] 
        for idx in self.params_index:
            name = idx['name']
            value_slot = idx['value_slot']
            opt_x0.append(self.params[name]['value'][value_slot]) 
        return np.array(opt_x0)



    def echo_params(self):
        print '='*80
        print 'Potential model parameters that will be optimzied:\n'
        for name,attr in self.params.iteritems(): 
            print name
            for i in range(attr['size']):
                print attr['value'][i], '  ',
                if not attr['fix'][i] and attr['lower_bound'][i] == None:
                    print   # print new line if only given value
                if attr['fix'][i]:
                    print 'fix'
                if attr['lower_bound'][i] != None:
                    print attr['lower_bound'][i], '  ',
                if attr['upper_bound'][i]:
                    print attr['upper_bound'][i]
            print 








    def compute(self):
        ks.KIM_API_model_compute(self.pkim)


    def get_coords(self):
        if self.km_coords is not None:
            return self.km_coords.copy()
        else:
            raise SupportError("forces")

    def get_energy(self):
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
            ks.KIM_API_model_destroy(self.pkim)
            ks.KIM_API_free(self.pkim)
        self.pkim = None



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




def init_KIMobjects(modelname, confs, initial_params):
    '''
    Wrapper function to instantiate multiple KIMobject class, one for each 
    configuration in the training set.
    '''
    kim_objects = []
    for c in confs:
        obj = KIMobject(modelname, c)
        obj.initialize()
        obj.map_opt_index(initial_params)
        obj.compute()
        kim_objects.append(obj)
    return kim_objects



if __name__ == '__main__':
   
    # test generate_kimstr()
    from training import TrainingSet
    tset = TrainingSet()
    #modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
    
    #tset.read('../tests/training_set')
    #tset.read('../tests/training_set/T150_training_1000.xyz')
    tset.read('../tests/config.txt_20x20')
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    #tset.read('../tests/training_set_Si.xyz')
    #modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
    configs = tset.get_configs()
    

    # initialize objects
    KIMobj = KIMobject(modelname, configs[0]) 
    KIMobj.initialize()

    KIMobj.compute()  
    print KIMobj.get_NBC_method()
    print KIMobj.get_energy() 
    print KIMobj.get_forces()[0:3]

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

#    ks.KIM_API_print(KIMobj.pkim)

    
