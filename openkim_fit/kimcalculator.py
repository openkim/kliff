from __future__ import division
import numpy as np
import kimservice as ks
import kimneighborlist as kimnl
from utils import generate_kimstr, write_extxyz
from modelparams import ModelParams
from neighbor import set_padding
import sys


class KIMcalculator:
    """ KIM calculator class that computes the energy, forces for a given configuration.

    Parameters
    ----------

    modelname: str
        the KIM Model upon which the KIM object is built

    opt_params: ModelParams object

    conf: Configuration object in which the atoms information are stored

    use_energy: bool
        whether to include energy in the prediction output

    """

    def __init__(self, modelname, opt_params, conf, use_energy=False):

        self.modelname = modelname
        self.opt_params = opt_params
        self.conf = conf
        self.use_energy = use_energy

        # initialize pointers for kim
        self.km_nparticles = None
        self.km_nspecies = None
        self.km_particle_spec_code = None
        self.km_particle_status = None
        self.km_coords = None

        # output of model
        self.km_energy = None
        self.km_forces = None
        self.km_cutoff = None
#        self.km_particleEnergy = None
#        self.km_virial = None
#        self.km_particleVirial = None
#        self.km_hessian = None

        # the KIM object
        self.pkim = None
        self.NBC = None

        # neighbor list
        self.uses_neighbors = None
        self.pad_image = None
        self.ncontrib = None
        self.need_padding_neigh = True

        #  parameters
        self.params = dict()


    def initialize(self):
        """ Initialize the KIM object for the configuration of atoms in self.conf."""

        # inquire information from the conf
        particle_spec = self.conf.get_species()
        species_set = set(particle_spec)
        self.km_nspecies =  np.array([len(species_set)]).astype(np.int32)
        self.km_nparticles = np.array([self.conf.get_num_atoms()]).astype(np.int32)
        self.km_coords = self.conf.get_coords()
        cell = self.conf.get_cell()
        PBC = self.conf.get_pbc()

        kimstr = generate_kimstr(self.modelname, cell, species_set)
        status, self.pkim = ks.KIM_API_init_str(kimstr, self.modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)

        # init model
        # memory for `cutoff' must be allocated and registered before calling model_init
        self.km_cutoff = np.array([0.])
        ks.KIM_API_set_data_double(self.pkim, "cutoff", self.km_cutoff)
        ks.KIM_API_model_init(self.pkim)

        # create padding atoms if NBC is pure
        self.NBC = self.get_NBC_method()
        if 'PURE' in self.NBC:
            self.ncontrib = self.km_nparticles[0]
            pad_coords, pad_spec, self.pad_image = set_padding(cell, PBC, particle_spec,
                self.km_coords, self.km_cutoff[0])

            self.km_coords = np.concatenate((self.km_coords, pad_coords))
            particle_spec = np.concatenate((particle_spec, pad_spec))
            npadding = len(pad_spec)
            self.km_nparticles[0] += npadding

            # set particle status (contributing or not)
            self.km_particle_status =np.concatenate((np.ones(self.ncontrib),
                np.zeros(npadding))).astype(np.int32)

            # assume all atoms need neigh, even padding atoms
            not_need_neigh = np.zeros(self.km_nparticles[0]).astype(np.int32)
            # turn off generating neighbors for those who do not need
            if not self.need_padding_neigh:
                not_need_neigh[self.ncontrib:] = 1

# NOTE for debug use only
#        write_extxyz(cell, particle_spec, self.km_coords, fname='check_set_padding2.xyz')

        # get species code for all the atoms
        self.km_particle_spec_code = []
        for i,s in enumerate(particle_spec):
            self.km_particle_spec_code.append(ks.KIM_API_get_species_code(self.pkim, s))
        self.km_particle_spec_code = np.array(self.km_particle_spec_code).astype(np.int32)

        # set KIM object pointers (input for KIM object)
        ks.KIM_API_set_data_int(self.pkim, "numberOfParticles", self.km_nparticles)
        ks.KIM_API_set_data_double(self.pkim, "coordinates", self.km_coords)
        ks.KIM_API_set_data_int(self.pkim, "numberOfSpecies", self.km_nspecies)
        ks.KIM_API_set_data_int(self.pkim, "particleSpecies", self.km_particle_spec_code)
        ks.KIM_API_set_data_int(self.pkim, "particleStatus", self.km_particle_status)

        # initialize energy and forces and register their KIM pointer (output of KIM object)
        self.km_energy = np.array([0.])
        self.km_forces = np.array([0.0]*(3*self.km_nparticles[0]))  # 3 for 3D
        ks.KIM_API_set_data_double(self.pkim, "energy", self.km_energy)
        ks.KIM_API_set_data_double(self.pkim, "forces", self.km_forces)

        # get parameter value in KIM object (can be updated through update_param)
        opt_param_names = self.opt_params.get_names()
        for name in opt_param_names:
            value = ks.KIM_API_get_data_double(self.pkim, name)
            size = self.opt_params.get_size(name)
            self.params[name] = {'size':size,'value':value}
        # this needs to be called before setting up neighborlist, since possibly
        # the cutoff may be changed through FREE_PARAM_ ...
        self._update_params()


        # set up neighbor list
        cell = self.conf.get_cell().flatten()
        if self.NBC == 'CLUSTER':
            self.uses_neighbors = False
        else:
            self.uses_neighbors = True

        if self.uses_neighbors == True:
            kimnl.nbl_initialize(self.pkim)
# NOTE PURE does not care about PBC, it is not used internally in the neighborlist code
            kimnl.nbl_set_cell(cell, PBC)
            if 'PURE' in self.NBC:
                kimnl.nbl_set_ghosts(not_need_neigh, 0)
            kimnl.nbl_build_neighborlist(self.pkim)




    def _update_params(self):
        """Update potential model parameters from ModelParams class to KIM object."""

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


    def assemble_padding_forces(self, forces):
        """
        Assemble forces on padding atoms back to contributing atoms.

        Parameters
        ----------

        forces: list
            forces of both contributing and padding atoms

        Return
        ------
            forces on contributing atoms
        """

        forces = np.reshape(forces, (len(forces)//3, 3))
        pad_forces = forces[self.ncontrib:]
        for i in range(self.ncontrib):
            # idx: the indices of padding atoms that are images of contributing atom i
            idx = np.where(np.array(self.pad_image) == i)
            forces[i] += np.sum(pad_forces[idx], axis=0)
        # only return forces of contributing atoms
        return np.reshape(forces[:self.ncontrib], 3*self.ncontrib)


    def compute(self):
        ks.KIM_API_model_compute(self.pkim)

    def get_prediction(self):
        self._update_params()
        self.compute()

        if self.km_energy is not None:
            energy = self.km_energy.copy()[0]
        else:
            raise SupportError("energy")
        if self.km_forces is not None:
            forces = self.km_forces.copy()
        else:
            raise SupportError("force")

        if 'PURE' in self.NBC:
            forces = self.assemble_padding_forces(forces)

        if self.use_energy:
            return np.concatenate(([energy], forces))
        else:
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
            forces = self.km_forces.copy()
            if 'PURE' in self.NBC:
                forces = self.assemble_padding_forces(forces)

            return forces
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


