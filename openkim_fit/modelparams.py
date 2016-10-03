from __future__ import print_function
import numpy as np
from collections import OrderedDict
import kimservice as ks
from error import InputError
from utils import generate_dummy_kimstr
from utils import remove_comments 

class ModelParams():
    '''
    Class of the potential model parameters. It will interact with optimizer to
    provide initial guesses of parameters and receive updated paramters. Also, 
    prediction tests will inqure updated parameters from this class.
    '''

    def __init__(self, modelname):
        '''
        Parameters:
       
        modelname: KIM model name
        '''
        self._modelname = modelname
        self._avail_params = OrderedDict()
        self._params = OrderedDict()
        self._params_index = []
        self._pkim = None
        self._cutoff = None
        # inquire KIM for the available parameters
        self._get_avail_params()


    def read(self,fname):
        '''
        For a given model parameter, one or multiple initial values may be required,
        and each must be given in a new line. For each line, the initial guess value 
        is mandatory, where 'KIM' (case insensitive) can be given to use the value 
        from the KIM model. Optionally, "fix" can be followed not to optimize this 
        parameters, or lower and upper bounds can be given to limit the parameter 
        in the range.  The following are valid input examples. 
        
        Examples:
        
        PARAM_FREE_A
        KIM
        1.1

        PARAM_FREE_B
        KIM  fix
        1.1  fix

        PARAM_FREE_C
        KIM  0.1  2.1
        1.0  0.1  2.1
        2.0  fix 

        Params:

        fname (str), name of the file where the parameters to optimize are listed.
        '''

        with open (fname, 'r') as fin:
            lines = fin.readlines()
            lines = remove_comments(lines)
        num_line = 0
        while num_line < len(lines):
            line = lines[num_line].strip(); num_line+=1
            if line in self._params:
                raise InputError('line: {} file: {}. Parameter {} already '
                                 'set.'.format(num_line, fname, line))
            if line not in self._avail_params:
                if 'PARAM_FREE' in line:
                    raise InputError('line: {} file: {}. Parameter {} not supported by '
                                     'the potential model.'.format(num_line,fname,line))
                else:
                    continue
            name = line
            size = self._avail_params[name]['size']
            param_lines = [name] 
            for j in range(size):
                param_lines.append(lines[num_line].split())
                num_line += 1
            self.set_param(param_lines)

    def set_param(self, lines):
        '''
        Set parameters that will be optimized in the ModelParams object. This is
        an alternative of the read method of this class.  The name of the parameter
        should be given as the first entry of a list (or tuple), and then each data 
        line should be given in in a list.

        Example:
            param_A = ['PARAM_FREE_A',
                       ['kim', 0, 20],
                       [2.0, 'fix'],
                       [2.2, 1.1, 3.3]
                      ]
            instance_of_this_class.set_param(param_A)
        '''
        name = lines[0].strip()
        if name in self._params:
            raise InputError('Parameter {} already set.'.format(name))
        if name not in self._avail_params:
            raise InputError('Parameter {} not supported by the potential model.'.format(name))
        size = self._avail_params[name]['size']
        if len(lines)-1 != size:
            raise InputError('Incorrect number of data lines for paramter {}.'.format(name))
        tmp_dict = {'size':        size,
                    'value':       np.array([None for i in range(size)]), 
                    'use-kim':     np.array([False for i in range(size)]), 
                    'fix':         np.array([False for i in range(size)]), 
                    'lower_bound': np.array([None for i in range(size)]), 
                    'upper_bound': np.array([None for i in range(size)])}
        self._params[name] = tmp_dict 
        for j in range(size):
            line = lines[j+1]
            num_items = len(line)
            if num_items == 1:
                self._read_1_item(name, j, line)
            elif num_items == 2:
                self._read_2_item(name, j, line)
            elif num_items == 3:
                self._read_3_item(name, j, line)
            else:
                raise InputError('More than 3 iterms listed at data line '
                                 '{} for parameter {}.'.format(j+1, name))
            self._check_bounds(name)
        self._set_param_index(name)

    def echo_avail_params(self):
        '''
        Echo the adjustable parameters to stdout.
        '''
        print()
        print('='*80)
        print('The following potential model parameters are available to fit. Include')
        print('the names and the initial guesses (optionally, lower and upper bounds)')
        print('of the parameters that you want to optimize in the input file.')
        print()
        for name,attr in self._avail_params.iteritems(): 
            print('name: ', name)
#            print('rank: ', attr['rank'])
#            print('shape:', attr['shape'])
            print('size: ', attr['size'])
            print('data: ', attr['value'])
            print()


    def echo_params(self):
        print()
        print('='*80)
        print('Potential model parameters that will be optimzied:')
        print()
        for name,attr in self._params.iteritems(): 
            print (name)
            for i in range(attr['size']):
                print(attr['value'][i], end='  ')
                if not attr['fix'][i] and attr['lower_bound'][i] == None:
                    print()   # print new line if only given value
                if attr['fix'][i]:
                    print('fix')
                if attr['lower_bound'][i] != None:
                    print(attr['lower_bound'][i], end='  ')
                if attr['upper_bound'][i]:
                    print(attr['upper_bound'][i])
            print() 

#NOTE, the following will echo KIM, if the initial params is from KIM
#
#    def echo_params(self):
#        print()
#        print('='*80)
#        print('Potential model parameters that will be optimzied:')
#        print()
#        for name,attr in self._params.iteritems(): 
#            print (name)
#            for i in range(attr['size']):
#                if attr['use-kim'][i]:
#                    print('KIM', end='  ')
#                else:
#                    print(attr['value'][i], end='  ')
#                if not attr['fix'][i] and attr['lower_bound'][i] == None:
#                    print()   # print new line if only given value
#                if attr['fix'][i]:
#                    print('fix')
#                if attr['lower_bound'][i] != None:
#                    print(attr['lower_bound'][i], end='  ')
#                if attr['upper_bound'][i]:
#                    print(attr['upper_bound'][i])
#            print() 
#

    def set_cutoff(self, cutoff):
        self._cutoff = cutoff
    
    def get_names(self):
        return np.array(self._params.keys()).copy()
#    def get_rank(self, name):
#        return self._params[name]['rank']
#    def get_shape(self, name):
#        return self._params[name]['shape']
    def get_size(self, name):
        return self._params[name]['size'].copy()
    def get_value(self, name):
        return self._params[name]['value'].copy()
#    def get_lower_bound(self, name):
#        return self._params[name]['lower_bound'].copy()
#    def get_upper_bound(self, name):
#        return self._params[name]['upper_bound'].copy()
#    def get_fix(self, name):
#        return self._params[name]['fix'].copy()

    def get_cutoff(self):
        return self._cutoff

    def update_params(self, opt_x):
        '''
        Update parameter values from optimzier. 
        '''
        for i,val in enumerate(opt_x):
            name = self._params_index[i]['name']
            value_slot = self._params_index[i]['value_slot']
            self._params[name]['value'][value_slot] = val

    def get_x0(self):
        '''
        Nest all parameter values (except the fix ones) to a list. And this will
        be fed to the optimizer as the starting parameters.
        '''
        opt_x0 = [] 
        for idx in self._params_index:
            name = idx['name']
            value_slot = idx['value_slot']
            opt_x0.append(self._params[name]['value'][value_slot]) 
        return np.array(opt_x0)

    def _get_avail_params(self):
        '''
        Inqure the potential model to get all the potential parameters whoes values 
        are allowed to adjust. Namely, the "PARAM_FREE" in the model's descriptor file.
        '''
        kimstr = generate_dummy_kimstr(self._modelname)
        status, self._pkim = ks.KIM_API_init_str(kimstr, self._modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self._modelname)
        # set dummy numberOfSpeces and numberOfParticles to 1
        ks.KIM_API_allocate(self._pkim, 1, 1) 
        ks.KIM_API_model_init(self._pkim)
        N_free_params = ks.KIM_API_get_num_free_params(self._pkim)
        for i in range(N_free_params):
            name = ks.KIM_API_get_free_parameter(self._pkim, i)
            rank = ks.KIM_API_get_rank(self._pkim, name)
            shape = ks.KIM_API_get_shape(self._pkim, name)
            value = ks.KIM_API_get_data_double(self._pkim, name)
            if rank == 0:
                size = 1
            else:
                size = np.prod(shape) 
            self._avail_params[name] = {'rank':rank, 'shape':shape, 
                                       'size':size, 'value':value}
        # cutoff
        cutoff = ks.KIM_API_get_data_double(self._pkim, 'cutoff')
        self._cutoff = cutoff[0]
        

    def _read_1_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])


    def _read_2_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        if line[1].lower() == 'fix':
            self._params[name]['fix'][j] = True
        else:
            raise InputError('Data at line {} of {} corrupted.\n'.format(j+1, name))


    def _read_3_item(self, name, j, line):
        self._read_1st_item(name, j, line[0])
        try:
            self._params[name]['lower_bound'][j] = float(line[1])
            self._params[name]['upper_bound'][j] = float(line[2])
        except ValueError as err:
            raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))


    def _read_1st_item(self, name, j, first):
        if type(first)==str and first.lower() == 'kim':
            self._params[name]['use-kim'][j] = True
            model_value = self._avail_params[name]['value']
            self._params[name]['value'][j] = model_value[j] 
        else:
            try:
                self._params[name]['value'][j] = float(first) 
            except ValueError as err:
                raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

        
    def _check_bounds(self, name):
        '''
        Check whether the initial guess of a paramter is within its lower and
        upper bounds.
        '''
        attr = self._params[name]
        for i in range(attr['size']):
            lower_bound = attr['lower_bound'][i] 
            upper_bound = attr['upper_bound'][i] 
            if lower_bound != None:
                value = attr['value'][i]
                if value < lower_bound or value > upper_bound:
                    raise InputError('Initial guess at line {} of parameter {} is '
                                     'out of bounds.\n'.format(i+1, name))


    def _set_param_index(self, name):
        '''
        Check whether a specific data value of a parameter will be optimized or 
        not (by checking its "fix" attribute). If yes, include it in the index
        list.
        
        Given a parameter at its values such as:
        
        PARAM_FREE_B
        1.1 
        2.2  fix
        4.4  3.3  5.5 
      
        the first slot (1.1) and the third slot (4.4) will be included in the 
        _params_index, and later be optimized. 
        '''
        size = self._params[name]['size']  
        fix  = self._params[name]['fix']  
        for i in range(size):
            if not fix[i]:
                tmp_idx = {'name':name, 'value_slot':i}
                self._params_index.append(tmp_idx)


    def __del__(self):
        ''' Garbage collects the KIM API objects automatically '''
        if self._pkim:
            ks.KIM_API_model_destroy(self._pkim)
            ks.KIM_API_free(self._pkim)
        self._pkim = None




if __name__ == '__main__':
    modelname = 'Pair_Lennard_Jones_Truncated_Nguyen_Ar__MO_398194508715_000'
    modelname = 'EDIP_BOP_Bazant_Kaxiras_Si__MO_958932894036_001'
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'
    
    # create a tmp input file
    lines=['PARAM_FREE_A']
    lines.append('kim 0 20')
    lines.append('2.0 fix')
    lines.append('2.0 fix')
    lines.append('PARAM_FREE_p')
    lines.append('kim 0 20')
    lines.append('2.0  1.0  3.0')
    lines.append('2.0 fix')
    fname = 'test_params.txt'
    with open(fname, 'w') as fout:
        for line in lines:
            fout.write(line+'\n')


    att_params = ModelParams(modelname)
    att_params._get_avail_params()
    att_params.echo_avail_params()
   
    att_params.read(fname)

    param_A = ['PARAM_FREE_A',
               ['kim', 0, 20],
               [2.0, 'fix'],
               [2.2, 1.1, 3.3]]
#    att_params.set_param(param_A)
   
    param_B = ('PARAM_FREE_B',
               ('kim', 0, 20),
               (2.0, 'fix'),
               (2.2, 1.1, 3.3))
    att_params.set_param(param_B)

    att_params.echo_params()    

    print( att_params.get_value('PARAM_FREE_A'))
    print( att_params.get_size('PARAM_FREE_A'))
    print('cutoff', att_params.get_cutoff())
    att_params.set_cutoff(6.0)
    print('cutoff after set', att_params.get_cutoff())




