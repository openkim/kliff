import sys
import numpy as np
from collections import OrderedDict
import kimservice as ks
from error import InputError
from utils import generate_dummy_kimstr

class ModelParams():
    '''
    Class to handle the model parameters that are read in from the input file.
    For a model paramter, a number of "size" initial values are required, each 
    given in a new line. For each line, the initial guess value is mandatory,
    where 'KIM' (case insensitive) can be given to use the value from the KIM
    model. Optionally, "fix" can be followed not to optimize this parameters, 
    or lower and upper bounds can be given to limit the parameter in the range.
    The following are valid input examples. 
    
    Examples:
    --------
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
    '''
    def __init__(self, lines, modelname):
        '''
        Parameters:
        
        lines: the input data (in the form of list) that contains the parameters info.
        '''
        self.lines = lines
        self.modelname = modelname
        self.avail_params = dict()
        self.opt_params = OrderedDict()
        self.pkim = None

 
    def get_avail_params(self):
        '''
        Inqure the potential model to get all the potential parameters whoes values 
        are allowed to adjust. Namely, the "PARAM_FREE" in the model's descriptor file.
        '''
        kimstr = generate_dummy_kimstr(self.modelname)
        status, self.pkim = ks.KIM_API_init_str(kimstr, self.modelname)
        if ks.KIM_STATUS_OK != status:
            ks.KIM_API_report_error('KIM_API_init', status)
            raise InitializationError(self.modelname)
        # set dummy numberOfSpeces and numberOfParticles to 1
        ks.KIM_API_allocate(self.pkim, 1, 1) 
        ks.KIM_API_model_init(self.pkim)

#NOTE modifications needed to call KIM_API_get_free_params to get the names
        free_params = ['PARAM_FREE_A', 'PARAM_FREE_B', 'PARAM_FREE_p'] 
        for name in free_params:
#NOTE modifications needed to call get_rank, get_shape and get_data to set them
            rank = 1
            shape = 3
            value = ks.KIM_API_get_data_double(self.pkim, name)
            size = np.prod(shape) 
            self.avail_params[name] = {'rank':rank, 'shape':shape, 
                                       'size':size, 'value':value}


    def echo_avail_params(self):
        '''
        Echo the adjustable parameters to stdout.
        '''
        print
        print '='*80
        print ('The following potential model parameters are available to fit. Include the\n'
               'names and the initial guesses (optionally, lower and upper bounds) of the\n'
               'parameters that you want to optimize in the input file.\n')
        for name,attr in self.avail_params.iteritems(): 
            print 'name: ', name
            print 'rank: ', attr['rank']
            print 'shape:', attr['shape']
            print 'size: ', attr['size']
            print 'data: ', attr['value']
            print
        sys.exit(0)



    def read(self):
        avail_names = self.avail_params.keys()
        lines = self.lines 
        num_line = 0
        while num_line < len(lines):
            line = lines[num_line].strip(); num_line+=1
            if line not in avail_names:
                if 'PARAM_FREE' in line:
                    raise InputError('Parameter {} is not supported by the potential '
                                      'model.'.format(line))
                else:
                    continue
            else:
                name = line
                rank = self.avail_params[name]['rank']
                shape = self.avail_params[name]['shape']
                size = self.avail_params[name]['size']
                tmp_dict = {'rank':        rank,
                            'shape':       shape,
                            'size':        size,
                            'value':       [None for i in range(size)], 
                            'use-kim':     [False for i in range(size)], 
                            'fix':         [False for i in range(size)], 
                            'lower_bound': [None for i in range(size)], 
                            'upper_bound': [None for i in range(size)]}
                self.opt_params[name] = tmp_dict 
                for j in range(size):
                    line = lines[num_line].strip(); num_line+=1
                    line = line.split()
                    num_items = len(line)
                    if num_items == 1:
                        self.read_1_item(name, j, line)
                    elif num_items == 2:
                        self.read_2_item(name, j, line)
                    elif num_items == 3:
                        self.read_3_item(name, j, line)
                    else:
                        raise InputError('More than 3 iterms listed at data line '
                                         '{} for {}.'.format(j+1, name))
        # check whether initial guess is within bounds 
        self.check_bounds()

    def read_1_item(self, name, j, line):
        self.read_1st_item(name, j, line[0])

    def read_2_item(self, name, j, line):
        self.read_1st_item(name, j, line[0])
        if line[1].lower() == 'fix':
            self.opt_params[name]['fix'][j] = True
        else:
            raise InputError('Data at line {} of {} corrupted.\n'.format(j+1, name))

    def read_3_item(self, name, j, line):
        self.read_1st_item(name, j, line[0])
        try:
            self.opt_params[name]['lower_bound'][j] = float(line[1])
            self.opt_params[name]['upper_bound'][j] = float(line[2])
        except ValueError as err:
            raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

    def read_1st_item(self, name, j, first):
        if first.lower() == 'kim':
            self.opt_params[name]['use-kim'][j] = True
        else:
            try:
                self.opt_params[name]['value'][j] = float(first) 
            except ValueError as err:
                raise InputError('{}.\nData at line {} of {} corrupted.\n'.format(err, j+1, name))

        
    def check_bounds(self):
        '''
        Check whether the initial guess of a paramter is within its lower and upper bounds.
        '''
        for name,attr in self.opt_params.iteritems():
            model_value = self.avail_params[name]['value']
            for i in range(attr['size']):
                lower_bound = attr['lower_bound'][i] 
                if lower_bound != None:
                    upper_bound = attr['upper_bound'][i] 
                    read_value = attr['value'][i]
                    if attr['use-kim'][i]:
                        read_value = model_value[i]
                    if read_value<lower_bound or read_value>upper_bound:
                        raise InputError('Initial guess at line {} of {} is out of '
                                         'bounds.\n'.format(i+1, name))


    def echo_opt_params(self):
        print '='*80
        print 'Potential model parameters that will be optimzied:\n'
        for name,attr in self.opt_params.iteritems(): 
            print name
            for i in range(attr['size']):
                if attr['use-kim'][i]:
                    print 'KIM   ',
                else:
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



    def get_names(self):
        return self.opt_params.keys()
    def get_rank(self, name):
        return self.opt_params[name]['rank']
    def get_shape(self, name):
        return self.opt_params[name]['shape']
    def get_size(self, name):
        return self.opt_params[name]['size']
    def get_value(self, name):
        return self.opt_params[name]['value']
    def get_lower_bound(self, name):
        return self.opt_params[name]['lower_bound']
    def get_upper_bound(self, name):
        return self.opt_params[name]['upper_bound']
    def get_fix(self, name):
        return self.opt_params[name]['fix']


    def __del__(self):
        ''' Garbage collects the KIM API objects automatically '''
        if self.pkim:
            ks.KIM_API_model_destroy(self.pkim)
            ks.KIM_API_free(self.pkim)
        self.pkim = None




if __name__ == '__main__':
    modelname = 'Three_Body_Stillinger_Weber_MoS__MO_000000111111_000'

    # test FreeParam class
#    free_params = FreeParams(modelname)
#    free_params.inquire_free_params()
#    free_params.echo()
    
    lines=['PARAM_FREE_A']
    lines.append('kim 0 20')
    lines.append('2.0 fix')
    lines.append('2.0 fix')
    lines.append('PARAM_FREE_p')
    lines.append('kim 0 20')
    lines.append('2.0  1.0  3.0')
    lines.append('2.0 fix')


    att_params = ModelParams(lines, modelname)
    att_params.get_avail_params()
    #att_params.echo_avail_params()
    att_params.read()
    att_params.echo_opt_params()    
    #print lines






