import numpy as np
import kimservice as ks
import kimneighborlist as kimnl
from utils import generate_kimstr
from utils import checkIndex
from modelparams import ModelParams

import sys
class KIMcalculator:
    '''
    KIM calculator class that can compute the energy, forces, and stresses for
    a given configuration.
    '''

    def __init__(self, modelname, opt_params, conf):
        '''
        Creates a KIM calculator for a given potential model.

        Parameters

        modelname: str
            the KIM Model upon which the KIM object is built

        params: instance of class ModelParams

        conf: Config object in which the atoms information are stored
        Returns

        out: KIM object
        '''

        # class members
        self.modelname = modelname
        self.opt_params = opt_params
        self.conf = conf

        # initialize pointers for kim
        self.km_nparticles = None
        self.km_nspecies = None
        self.km_particle_spec_code = None
        self.km_coords = None

        # output of model
        self.km_energy = None
        self.km_forces = None
        self.km_cutoff = None           # this is needed init model; but not used
#        self.km_particleEnergy = None
#        self.km_virial = None
#        self.km_particleVirial = None
#        self.km_hessian = None

        # the KIM object
        self.pkim = None
        self.uses_neighbors = None

        #  parameters
        self.params = dict()



    def initialize(self):
        ''' Initialize the KIM object for the configuration of atoms in self.conf.'''

        # inquire information from the conf
        particle_species = self.conf.get_species()
        species = set(particle_species)
        self.km_nspecies =  np.array([len(species)]).astype(np.int32)
        self.km_nparticles = np.array([self.conf.get_num_atoms()]).astype(np.int32)
        self.km_coords = self.conf.get_coords()
        cell = self.conf.get_cell()

        kimstr = generate_kimstr(self.modelname, cell, species)
        status, self.pkim = ks.KIM_API_init_str(kimstr, self.modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)

#        # get species code for all the atoms
        self.km_particle_spec_code = []
        for i,s in enumerate(particle_species):
            self.km_particle_spec_code.append(ks.KIM_API_get_species_code(self.pkim, s))
        self.km_particle_spec_code = np.array(self.km_particle_spec_code).astype(np.int32)

        # set KIM object pointers (input for KIM object)
        ks.KIM_API_set_data_int(self.pkim, "numberOfParticles", self.km_nparticles)
        ks.KIM_API_set_data_double(self.pkim, "coordinates", self.km_coords)
        ks.KIM_API_set_data_int(self.pkim, "numberOfSpecies", self.km_nspecies)
        ks.KIM_API_set_data_int(self.pkim, "particleSpecies", self.km_particle_spec_code)

        # initialize energy and forces and register their KIM pointer (output of KIM object)
        self.km_energy = np.array([0.])
        self.km_forces = np.array([0.0]*(3*self.km_nparticles[0]))  # 3 for 3D
        ks.KIM_API_set_data_double(self.pkim, "energy", self.km_energy)
        ks.KIM_API_set_data_double(self.pkim, "forces", self.km_forces)

        # init model
        # memory for `cutoff' must be allocated and registered before calling model_init
        self.km_cutoff = np.array([0.])
        ks.KIM_API_set_data_double(self.pkim, "cutoff", self.km_cutoff)
        ks.KIM_API_model_init(self.pkim)

#NOTE we may need numberOfcontributingAtoms to use half list
# if we want to use MIOPBC, we need to add something below  box side length see potfit

        # get parameter value in KIM object (can be updated through update_param)
        opt_param_names = self.opt_params.get_names()
        for name in opt_param_names:
            value = ks.KIM_API_get_data_double(self.pkim, name)
            size = self.opt_params.get_size(name)
            self.params[name] = {'size':size,'value':value}
        # this needs to be called before setting up neighborlist, since possibly
        # the cutoff may be changed through FREE_PARAM_ ...
        self.update_params()

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


    def update_params(self):
        '''
        Update potential model parameters from ModelParams class to KIM object.
        '''
        for name,attr in self.params.iteritems():
            new_value = self.opt_params.get_value(name)
            size  = attr['size']
            value = attr['value']
            for i in range(size):
                value[i] = new_value[i]
        ks.KIM_API_model_reinit(self.pkim)

#NOTE
#But we may want to move KIM_API_model_destroy to __del__

# this may not be needed, since the the __del__ will do the free automatically
    def free_kim(self):
        if self.uses_neighbors:
            kimnl.nbl_cleanup(self.pkim)
        ks.KIM_API_model_destroy(self.pkim)
        ks.KIM_API_free(self.pkim)
        self.pkim = None


    def compute(self):
        ks.KIM_API_model_compute(self.pkim)


    def get_prediction(self):
        self.update_params()
        self.compute()

        if self.km_energy is not None:
            energy = self.km_energy.copy()[0]
        else:
            raise SupportError("energy")
        if self.km_forces is not None:
            forces = self.km_forces.copy()
        else:
            raise SupportError("force")
#        return np.concatenate(([energy], forces))
# NOTE we only return forces here, may be modified
        return forces


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


#
#
#def init_KIMobjects(modelname, confs, initial_params):
#    '''
#    Wrapper function to instantiate multiple KIMobject class, one for each
#    configuration in the training set.
#    '''
#    kim_objects = []
#    for c in confs:
#        obj = KIMobject(modelname, c)
#        obj.initialize()
#        obj.map_opt_index(initial_params)
#        obj.compute()
#        kim_objects.append(obj)
#    return kim_objects
#
#

