import kimservice as ks
import kimneighborlist as kimnl
import 

def generate_kimstr(modelname, Tset):
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
    # code does not matter, so just give it 0 
    kimstr += 'PARTICLE_SPECIES:\n'
    kimstr += '# Symbol/name    Type    code\n'
    species = Tset.get_unique_species()
    for s in species:
        kimstr += s+'  spec    0\n.' 

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
    if (checkIndex(kimmdl, 'virial') >= 0 or checkIndex(km_pmdl, 'process_dEdr') >=0):
        kimstr += 'virial  double  energy  [6]\n'
    if (checkIndex(kimmdl, 'particleVirial') >= 0 or checkIndex(km_pmdl, 'process_dEdr') >=0):
        kimstr += 'particleVirial  double  energy  [numberOfParticles,6]\n'
    if (checkIndex(kimmdl, 'hessian') >= 0 or
	   (checkIndex(kimmdl, 'process_dEdr') >= 0 and checkIndex(km_pmdl, 'process_d2Edr2') >= 0)):
        kimstr += 'hessian  double  pressure  [numberOfParticles,numberOfParticles,3,3]\n'

    # free KIM object
    ks.KIM_API_model_destroy(kimmdl)
    ks.KIM_API_free(kimmdl)

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



if __name__ == '__main__':
    generate_kimstr()
